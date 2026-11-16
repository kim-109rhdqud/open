"""전부 자동: 수집(별도 스레드) → 라벨 → (라벨이 충분히 쌓이면) 합성·학습·검사 → 로봇 배포 를 반복.

멈추려면 Ctrl-C. 다시 켜면 하던 데서 이어 간다(판정한 사진·모델 버전은 work/ 에 남아 있음).
경기 중에는 돌리지 않는다(연습 시간 전용).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import deploy as deploy_mod
from . import train as train_mod
from .autolabel import AutoLabeler, frame_files, sessions
from .claude import ClaudeVerifier
from .collect import Collector

log = logging.getLogger("pipeline")


class Pipeline:
    def __init__(self, cfg: Dict, work: Path, code: str = "", claude_key: str = "",
                 watch: Optional[List[Path]] = None, collect: bool = True, do_train: bool = True,
                 do_deploy: Optional[bool] = None, fetcher=None) -> None:
        self.cfg, self.work = cfg, work
        self.stop = threading.Event()
        self.roots = [work / "frames"] + list(watch or [])
        self.collector: Optional[Collector] = None
        if collect:
            if not cfg["server"]["url"] or not code:
                raise SystemExit("수집하려면 서버 주소(MAICON_SERVER_URL 또는 local.yaml)와 임무코드가 필요하다")
            kw = {"fetcher": fetcher} if fetcher else {}
            self.collector = Collector(cfg, code, work / "frames", **kw)
        verifier = None
        if claude_key and cfg["claude"].get("enabled", True):
            c = cfg["claude"]
            verifier = ClaudeVerifier(claude_key, c["model"], float(c["timeout_s"]), int(c["max_side"]))
        self.labeler = AutoLabeler(cfg, work, verifier)
        self.do_train = do_train
        self.do_deploy = bool(cfg["deploy"].get("auto", True)) if do_deploy is None else do_deploy
        self.state_path = work / "state" / "pipeline.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.last_train: Optional[Dict] = None
        self.last_deploy: Optional[Dict] = None
        log.info("라벨 방식: %s", "OpenCV + Claude 검증" if verifier else "OpenCV 만 (Claude 키 없음)")

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")

    def status(self) -> Dict:
        n_frames = sum(len(frame_files(s)) for s in sessions(self.roots))
        cur = train_mod.current(self.work)
        dep_p = self.work / "models" / "deployed.json"
        dep = json.loads(dep_p.read_text(encoding="utf-8")) if dep_p.exists() else None
        st = {"time": time.strftime("%H:%M:%S"), "frames": n_frames, "labels": self.labeler.counts(),
              "model": cur, "deployed": dep, "last_train": self.last_train,
              "collector": None if not self.collector else {
                  "saved": self.collector.saved, "fails": self.collector.fails,
                  "last_error": self.collector.last_error, "stopped": self.collector.stopped_reason}}
        (self.work / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        return st

    @staticmethod
    def line(st: Dict) -> str:
        lb = st["labels"]
        m = st["model"]
        d = st["deployed"]
        s = f"[{st['time']}] 사진 {st['frames']} | 라벨 ok {lb['ok']} / 버림 {lb['rejected']}"
        s += f" | 모델 {m['version']} mAP50 {m['map50']}" if m else " | 모델 없음"
        if d:
            s += f" | 로봇 {d.get('version')} {'OK' if d.get('ok') else '실패'}" + (f" {d.get('ms')}ms" if d.get("ok") else "")
        c = st.get("collector")
        if c and (c["stopped"] or (c["fails"] and c["last_error"])):
            s += f" | 수집 {c['stopped'] or '실패 ' + str(c['fails']) + '번: ' + c['last_error']}"
        return s

    def cycle(self) -> None:
        self.labeler.process(self.roots)
        ok = self.labeler.counts()["ok"]
        t = self.cfg["train"]
        last = self.state.get("trained_at_ok")
        if self.do_train and ok >= int(t["min_labels"]) and (last is None or ok - last >= max(
                int(t["retrain_every"]), float(t.get("retrain_growth", 0)) * last)):
            self.state["trained_at_ok"] = ok
            self._save()
            try:
                self.last_train = train_mod.train_once(self.work, self.cfg)
            except Exception as e:
                log.exception("학습 오류")
                self.last_train = {"accepted": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            if self.last_train.get("accepted") and self.do_deploy:
                self.last_deploy = deploy_mod.deploy(self.work, self.cfg)
        elif self.do_deploy:
            # 지난번 배포가 실패했으면(로봇 꺼짐 등) 주기마다 다시
            cur = train_mod.current(self.work)
            dep_p = self.work / "models" / "deployed.json"
            dep = json.loads(dep_p.read_text(encoding="utf-8")) if dep_p.exists() else None
            if cur and (not dep or not dep.get("ok") or dep.get("version") != cur["version"]):
                if time.time() - self.state.get("deploy_try", 0) > 60:
                    self.state["deploy_try"] = time.time()
                    self.last_deploy = deploy_mod.deploy(self.work, self.cfg)

    def run(self, max_cycles: int = 0) -> None:
        th = None
        if self.collector:
            th = threading.Thread(target=self.collector.run, args=(self.stop,), daemon=True)
            th.start()
        n = 0
        try:
            while not self.stop.is_set():
                self.cycle()
                print(self.line(self.status()), flush=True)
                n += 1
                if max_cycles and n >= max_cycles:
                    break
                self.stop.wait(float(self.cfg["pipeline"]["cycle_s"]))
        except KeyboardInterrupt:
            print("\n멈춤 (다시 켜면 이어서 함)")
        finally:
            self.stop.set()
            if th:
                th.join(timeout=10)
            self.status()
