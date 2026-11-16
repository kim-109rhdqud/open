#!/usr/bin/env python3
"""드론 UGV 검출 자동 학습 — 명령 모음.

  python run.py selftest              # 처음 한 번: 가짜 서버로 수집→라벨→학습→배포 전체가 이 PC에서 도는지(인터넷·GPU 확인)
  python run.py ping                  # 경기 서버에서 드론 사진 한 장 받아 보기
  python run.py setup-key             # 로봇 ssh 키 등록(비밀번호 한 번)
  python run.py pipeline              # ★ 전부 자동: 수집 + 라벨 + 학습 + 배포 반복 (Ctrl-C 로 멈춤, 다시 켜면 이어서)
  python run.py pipeline --watch D    # 다른 도구(uav_collect.py)가 모은 폴더 D 도 같이 학습
  python run.py status                # 지금 상태 한 줄
  python run.py label | train | deploy [--version v003]   # 단계별로 직접

임무코드: 환경변수 MAICON_CODE (없으면 물어봄).  Claude 키: ANTHROPIC_API_KEY (없으면 물어봄, Enter 면 OpenCV 만).
현장 값(서버 주소·로봇 IP 등)은 local.yaml 에 적는다(config.yaml 참고).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from pathlib import Path

from dronecv import config as C


def _logging(work: Path) -> None:
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(work / "pipeline.log", encoding="utf-8")])
    for noisy in ("ultralytics", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_pipeline(a, cfg, work):
    from dronecv.pipeline import Pipeline
    collect = not a.no_collect
    code = C.mission_code() if collect else ""
    key = C.claude_key(ask=cfg["claude"].get("enabled", True))
    watch = [Path(w) for w in (a.watch or [])]
    Pipeline(cfg, work, code, key, watch=watch, collect=collect, do_train=not a.no_train,
             do_deploy=False if a.no_deploy else None).run()
    return 0


def cmd_ping(a, cfg, work):
    from dronecv.drone import FetchError, fetch
    if not cfg["server"]["url"]:
        print("서버 주소가 없다: local.yaml 의 server.url 또는 MAICON_SERVER_URL")
        return 2
    try:
        body, hdr = fetch(cfg["server"]["url"], C.mission_code(), cfg["server"]["camera"], 5.0)
    except FetchError as e:
        print("실패:", e)
        return 1
    out = work / "ping.jpg"
    out.write_bytes(body)
    print(f"성공: {len(body)} bytes → {out}")
    for k, v in hdr.items():
        print(f"  {k}: {v}")
    return 0


def cmd_label(a, cfg, work):
    from dronecv.autolabel import AutoLabeler
    from dronecv.claude import ClaudeVerifier
    key = C.claude_key(ask=cfg["claude"].get("enabled", True))
    c = cfg["claude"]
    v = ClaudeVerifier(key, c["model"], float(c["timeout_s"]), int(c["max_side"])) if key and c.get("enabled", True) else None
    lab = AutoLabeler(cfg, work, v)
    lab.process([work / "frames"] + [Path(w) for w in (a.watch or [])])
    print(lab.counts())
    return 0


def cmd_train(a, cfg, work):
    from dronecv.train import train_once
    res = train_once(work, cfg)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res.get("accepted") else 1


def cmd_deploy(a, cfg, work):
    from dronecv.deploy import deploy
    res = deploy(work, cfg, a.version)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res.get("ok") else 1


def cmd_setup_key(a, cfg, work):
    from dronecv.deploy import setup_key
    return setup_key(cfg)


def cmd_status(a, cfg, work):
    p = work / "status.json"
    if not p.exists():
        print("아직 상태 없음 (pipeline 을 돌리면 생김)")
        return 1
    from dronecv.pipeline import Pipeline
    print(Pipeline.line(json.loads(p.read_text(encoding="utf-8"))))
    return 0


def cmd_selftest(a, cfg, work):
    """가짜 서버와 가짜 사진으로 전체 흐름. 실제 성능과는 무관(이 PC에서 도는지만)."""
    from dronecv.autolabel import AutoLabeler
    from dronecv.collect import Collector
    from dronecv.deploy import deploy
    from dronecv.fakeserver import FakeDroneServer
    from dronecv.train import train_once
    tw = work.parent / (work.name + "_selftest")
    shutil.rmtree(tw, ignore_errors=True)
    tw.mkdir(parents=True)
    scfg = C.load(overrides={
        "server": {"interval_s": 0.0},
        "label": {"save_review": True},
        "augment": {"per_real": 3, "max_total": 300},
        "train": {"imgsz": 320, "epochs": a.epochs, "patience": 50, "batch": 8, "workers": 0, "min_map50": 0.0},
        "deploy": {"mode": "local", "local_dir": str(tw / "robot_sim")}})
    srv = FakeDroneServer(code="SELFTEST").start()
    scfg["server"]["url"] = srv.url
    t0 = time.time()
    steps = []
    try:
        col = Collector(scfg, "SELFTEST", tw / "frames", session="selftest")
        col.run(threading.Event(), count=a.frames)
        steps.append(("수집", col.saved >= a.frames * 0.8, f"{col.saved}장"))
        lab = AutoLabeler(scfg, tw, None)
        lab.process([tw / "frames"])
        cnt = lab.counts()
        steps.append(("자동 라벨", cnt["ok"] >= a.frames * 0.5, f"ok {cnt['ok']} / 버림 {cnt['rejected']}"))
        res = train_once(tw, scfg, seed=0)
        steps.append(("학습·ONNX", bool(res.get("onnx_check", {}).get("ok")),
                      f"{res.get('version')} mAP50 {res.get('map50')} ({res.get('minutes')}분)"))
        dep = deploy(tw, scfg)
        steps.append(("배포·로봇 점검(로컬)", bool(dep.get("ok")), f"추론 {dep.get('ms')} ms, UGV {dep.get('n')}대"))
    finally:
        srv.stop()
    print("\n=== selftest ===")
    for name, ok, info in steps:
        print(f"{'통과' if ok else '실패'}  {name}: {info}")
    all_ok = len(steps) == 4 and all(s[1] for s in steps)
    print(f"{'전부 통과' if all_ok else '실패 있음'} ({(time.time() - t0) / 60:.1f}분), 결과 폴더 {tw}")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="드론 UGV 검출 자동 학습")
    ap.add_argument("--config", default=None, help="local.yaml 대신 쓸 파일")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("--watch", action="append", help="추가로 지켜볼 사진 폴더(여러 번 가능)")
    p.add_argument("--no-collect", action="store_true", help="서버에서 받지 않고 폴더만")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-deploy", action="store_true")
    sub.add_parser("ping")
    p = sub.add_parser("label")
    p.add_argument("--watch", action="append")
    sub.add_parser("train")
    p = sub.add_parser("deploy")
    p.add_argument("--version", default=None)
    sub.add_parser("setup-key")
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    work = C.work_dir(cfg)
    _logging(work)
    fn = {"pipeline": cmd_pipeline, "ping": cmd_ping, "label": cmd_label, "train": cmd_train, "deploy": cmd_deploy,
          "setup-key": cmd_setup_key, "status": cmd_status, "selftest": cmd_selftest}[a.cmd]
    return fn(a, cfg, work)


if __name__ == "__main__":
    sys.exit(main())
