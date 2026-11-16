"""학습 → 성적 검사 → ONNX 내보내기.

결과: work/models/v001/{best.pt, best.onnx, model.json, metrics.json}
검사를 통과한 것만 work/models/current.json 이 가리킨다(배포 대상). 이전 모델은 지우지 않는다.
검사: 시험용 실사진 mAP50 ≥ min_map50, 그리고 지금 모델보다 tolerance 넘게 나빠지지 않을 것.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from . import augment, dataset

log = logging.getLogger("train")


def current(work: Path) -> Optional[Dict]:
    p = work / "models" / "current.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _next_version(work: Path) -> str:
    vs = [int(p.name[1:]) for p in (work / "models").glob("v[0-9][0-9][0-9]") if p.name[1:].isdigit()]
    return f"v{(max(vs) + 1 if vs else 1):03d}"


def _device(dev):
    if str(dev) != "auto":
        return dev
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def check_onnx(onnx: Path, image: Optional[Path]) -> Dict:
    """로봇과 같은 방식(OpenCV DNN)으로 열어 한 장 돌려 본다."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import cv2
    import robot_check
    net, meta = robot_check.load(str(onnx))
    if image is None:
        return {"ok": True, "n": None}
    img = cv2.imread(str(image))
    t = time.time()
    dets = robot_check.detect(net, img, meta)
    return {"ok": True, "n": len(dets), "ms": round((time.time() - t) * 1000, 1)}


def train_once(work: Path, cfg: Dict, seed: Optional[int] = None) -> Dict:
    t = cfg["train"]
    (work / "models").mkdir(parents=True, exist_ok=True)
    n_synth = augment.synthesize(work, cfg, seed)
    data, n_train, n_val = dataset.build(work, cfg)
    if n_val == 0:
        return {"accepted": False, "reason": "시험용 사진 없음"}
    from ultralytics import YOLO
    cur = current(work)
    base = t["base_model"]
    if t.get("continue_from_current", True) and cur:
        pt = work / "models" / cur["version"] / "best.pt"
        if pt.exists():
            base = str(pt)
    ver = _next_version(work)
    dev = _device(t["device"])
    log.info("학습 %s 시작: 출발 %s, 학습 %d장(합성 %d), 시험 %d장, 장치 %s", ver, Path(base).name, n_train, n_synth, n_val, dev)
    t0 = time.time()
    model = YOLO(base)
    model.train(data=str(data), imgsz=int(t["imgsz"]), epochs=int(t["epochs"]), patience=int(t["patience"]),
                batch=int(t["batch"]), workers=int(t["workers"]), device=dev, project=str(work / "runs"),
                name=ver, exist_ok=True, verbose=False, plots=True, seed=0 if seed is None else seed,
                flipud=0.5, fliplr=0.5, degrees=10.0)
    run = work / "runs" / ver
    best = run / "weights" / "best.pt"
    if not best.exists():
        return {"accepted": False, "reason": "best.pt 없음", "version": ver}
    m = YOLO(str(best)).val(data=str(data), imgsz=int(t["imgsz"]), split="val", device=dev,
                            project=str(work / "runs"), name=f"{ver}_val", exist_ok=True, verbose=False, plots=False)
    map50, map5095 = float(m.box.map50), float(m.box.map)
    out = work / "models" / ver
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out / "best.pt")
    exported = YOLO(str(out / "best.pt")).export(format="onnx", opset=int(t["opset"]), imgsz=int(t["imgsz"]),
                                                  dynamic=False, simplify=True)
    onnx = out / "best.onnx"
    if Path(exported).resolve() != onnx.resolve():
        shutil.move(str(exported), onnx)
    meta = {"version": ver, "imgsz": int(t["imgsz"]), "names": ["ugv"], "conf": 0.25, "iou": 0.5,
            "opset": int(t["opset"]), "map50": round(map50, 4), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    sample = next(iter(sorted((work / "dataset" / "real" / "images").glob("*.jpg"))), None)
    try:
        chk = check_onnx(onnx, sample)
    except Exception as e:
        chk = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    if sample is not None:
        shutil.copy2(sample, out / "sample.jpg")
    prev = float(cur.get("map50", 0.0)) if cur else 0.0
    reason = "ok"
    if not chk.get("ok"):
        reason = f"ONNX 점검 실패: {chk.get('error')}"
    elif map50 < float(t["min_map50"]):
        reason = f"mAP50 {map50:.3f} < 기준 {t['min_map50']}"
    elif cur and map50 < prev - float(t["tolerance"]):
        reason = f"mAP50 {map50:.3f} 이 지금 모델 {prev:.3f} 보다 나쁨"
    res = {"version": ver, "map50": round(map50, 4), "map50_95": round(map5095, 4), "n_train": n_train,
           "n_val": n_val, "n_synth": n_synth, "minutes": round((time.time() - t0) / 60, 1),
           "onnx_check": chk, "previous": cur["version"] if cur else None, "accepted": reason == "ok", "reason": reason}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if res["accepted"]:
        (work / "models" / "current.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("학습 %s 통과: mAP50 %.3f (이전 %s)", ver, map50, cur["version"] if cur else "없음")
    else:
        log.warning("학습 %s 배포 안 함: %s", ver, reason)
    return res
