"""새 드론 사진 → 자동 라벨 (YOLO 형식, 클래스 하나 'ugv').

판정 규칙
  1) OpenCV 후보(배경 차분 + 안테나 색)
  2) Claude 키가 있으면: 번호 상자마다 UGV·팀·딱 맞는지 + 놓친 UGV 수를 받음
     - 놓친 UGV 가 있거나, UGV 인데 상자가 안 맞으면 그 사진은 버림
  3) Claude 가 없으면: 배경 차분 덩어리 중 안테나 색이 있는 것만 UGV
  4) robots_in_arena > 0 이면 정확히 그 수만큼 찾은 사진만 씀(연습 때 우리 로봇 1대 → 1개)
버린 사진도 이유와 함께 기록한다(work/labels.jsonl, work/review/).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import vision
from .vision import Candidate

log = logging.getLogger("label")


def frame_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*.jpg") if p.is_file())


def sessions(roots: List[Path]) -> List[Path]:
    out = []
    for r in roots:
        if not r.exists():
            continue
        if frame_files(r):
            out.append(r)
        out += [d for d in sorted(r.iterdir()) if d.is_dir() and frame_files(d)]
    return out


class AutoLabeler:
    def __init__(self, cfg: Dict, work: Path, verifier=None) -> None:
        self.cfg, self.lcfg, self.work = cfg, cfg["label"], work
        self.verifier = verifier
        self.real_img = work / "dataset" / "real" / "images"
        self.real_lbl = work / "dataset" / "real" / "labels"
        self.crops = work / "dataset" / "crops"
        self.bgdir = work / "backgrounds"
        self.review = work / "review"
        for d in (self.real_img, self.real_lbl, self.crops, self.bgdir, work / "state"):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = work / "state" / "labeled.json"
        st = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.done: Dict[str, str] = st.get("done", {})
        self.sizes: List[Tuple[int, int, int]] = [tuple(x) for x in st.get("sizes", [])]  # (mask_px, w, h)
        self.bg_built: Dict[str, int] = st.get("bg_built", {})
        self._bg: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 상태
    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": self.done, "sizes": self.sizes[-500:], "bg_built": self.bg_built},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def counts(self) -> Dict[str, int]:
        vals = list(self.done.values())
        return {"ok": vals.count("ok"), "rejected": sum(1 for v in vals if v != "ok"), "total": len(vals)}

    def area_range(self) -> Optional[Tuple[float, float]]:
        if len(self.sizes) < int(self.lcfg["size_learn_after"]):
            return None
        m = float(np.median([s[0] for s in self.sizes]))
        return 0.35 * m, 2.5 * m

    def size_hint(self) -> Optional[Tuple[int, int]]:
        if len(self.sizes) < int(self.lcfg["size_learn_after"]):
            return None
        return int(np.median([s[1] for s in self.sizes])), int(np.median([s[2] for s in self.sizes]))

    # ---------------------------------------------------------------- 배경
    def background(self, session: Path) -> Optional[np.ndarray]:
        key = session.name
        files = frame_files(session)
        if len(files) < int(self.lcfg["min_frames_for_bg"]):
            return None
        built = self.bg_built.get(key, 0)
        path = self.bgdir / f"{key}.png"
        if key in self._bg and len(files) - built < int(self.lcfg["bg_refresh"]):
            return self._bg[key]
        if key not in self._bg and path.exists() and len(files) - built < int(self.lcfg["bg_refresh"]):
            self._bg[key] = vision.read(path)
            return self._bg[key]
        n = int(self.lcfg["bg_frames"])
        idx = np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int)
        imgs = [vision.read(files[i]) for i in sorted(set(idx))]
        shapes = {im.shape for im in imgs if im is not None}
        if len(shapes) != 1:
            return None
        bg = vision.median_background(imgs)
        vision.write(path, bg, ".png")
        self._bg[key], self.bg_built[key] = bg, len(files)
        return bg

    # ---------------------------------------------------------------- 판정
    def decide(self, img: np.ndarray, cands: List[Candidate]) -> Tuple[List[Candidate], str, Dict]:
        expect = int(self.lcfg["robots_in_arena"])
        info: Dict = {"n_cand": len(cands)}
        if not cands:
            return [], "no_candidate", info
        if self.verifier is not None:
            res = self.verifier.verify(img, cands)
            info["claude"] = res
            if res["missed"] > 0:
                return [], "claude_missed", info
            ok: List[Candidate] = []
            for i, c in enumerate(cands, 1):
                b = res["boxes"].get(i)
                if not b or not b["is_ugv"]:
                    continue
                if not b["tight"]:
                    return [], "claude_not_tight", info
                if b["team"] in ("RED", "BLUE"):
                    c.team = b["team"]
                ok.append(c)
        else:
            ok = [c for c in cands if c.fg and c.team]
            if len([c for c in cands if c.team]) != len(ok):
                return [], "ambiguous_color", info
        if expect > 0 and len(ok) != expect:
            return [], f"count_{len(ok)}_not_{expect}", info
        if not ok:
            return [], "no_ugv", info
        return ok, "ok", info

    def _label_one(self, session: Path, f: Path, bg: np.ndarray) -> Tuple[str, str, Dict]:
        img = vision.read(f)
        rel = f"{session.name}/{f.name}"
        if img is None:
            return rel, "unreadable", {}
        cands = vision.candidates(img, bg, self.lcfg, self.area_range(), self.size_hint())
        try:
            ok, why, info = self.decide(img, cands)
        except Exception as e:  # Claude 오류 → 이번엔 보류(다음 주기에 다시)
            return rel, "retry", {"error": str(e)[:120]}
        stem = f"{session.name}__{f.stem}"
        if why == "ok":
            H, W = img.shape[:2]
            lines = []
            for k, c in enumerate(ok):
                x1, y1, x2, y2 = c.box
                lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                if c.mask is not None and c.fg:
                    crop = img[y1:y2, x1:x2]
                    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = cv2.dilate(c.mask, np.ones((3, 3), np.uint8))
                    vision.write(self.crops / f"{stem}_{k}_{c.team or 'NA'}.png", bgra, ".png")
                    with self._lock:
                        self.sizes.append((int((c.mask > 0).sum()), x2 - x1, y2 - y1))
            vision.write(self.real_img / f"{stem}.jpg", img)
            (self.real_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.lcfg.get("save_review", True):
            colors = [(0, 255, 0) if c in ok else (0, 0, 255) for c in cands]
            pic = vision.draw(img, cands, True, colors)
            cv2.putText(pic, why, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if why == "ok" else (0, 0, 255), 2, cv2.LINE_AA)
            vision.write(self.review / ("ok" if why == "ok" else "rejected") / f"{stem}.jpg", pic)
        info["teams"] = [c.team for c in ok]
        return rel, why, info

    def process(self, roots: List[Path], limit: int = 0) -> Dict[str, int]:
        """아직 판정 안 한 사진을 판정한다. 새로 ok 가 된 수 등을 돌려준다."""
        new = {"ok": 0, "rejected": 0, "retry": 0, "waiting_bg": 0}
        jobs = []
        for s in sessions(roots):
            pending = [f for f in frame_files(s) if f"{s.name}/{f.name}" not in self.done]
            if not pending:
                continue
            bg = self.background(s)
            if bg is None:
                new["waiting_bg"] += len(pending)
                continue
            jobs += [(s, f, bg) for f in pending]
        limit = limit or int(self.lcfg.get("max_per_cycle", 0))
        if limit and len(jobs) > limit:
            jobs = jobs[-limit:]  # 최신 사진부터(밀린 옛 사진은 다음 주기에)
        if not jobs:
            return new
        workers = int(self.cfg["claude"]["workers"]) if self.verifier is not None else 1
        t0 = time.time()
        with open(self.work / "labels.jsonl", "a", encoding="utf-8") as logf:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for rel, why, info in ex.map(lambda j: self._label_one(*j), jobs):
                    if why == "retry":
                        new["retry"] += 1
                        continue
                    self.done[rel] = why
                    new["ok" if why == "ok" else "rejected"] += 1
                    logf.write(json.dumps({"frame": rel, "result": why, **info}, ensure_ascii=False) + "\n")
        self._save()
        log.info("라벨: 새로 ok %d, 버림 %d, 보류 %d (%.1fs)", new["ok"], new["rejected"], new["retry"], time.time() - t0)
        return new
