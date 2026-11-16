"""드론 사진 계속 받아 저장. work/frames/<세션>/f_00001.jpg + frames.csv.

- 같은 촬영 시각(X-Captured-At)은 다시 저장하지 않는다.
- 로봇이 멈춰 있어 사진이 거의 같으면 건너뛴다(그래도 n장마다 1장은 저장).
- 임무코드는 화면·파일에 남기지 않는다.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .drone import HEADERS, FetchError, fetch

log = logging.getLogger("collect")


def _small_gray(jpeg: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)


class Collector:
    def __init__(self, cfg: Dict, code: str, frames_root: Path,
                 fetcher: Callable = fetch, session: Optional[str] = None) -> None:
        self.cfg, self.code, self.fetcher = cfg, code, fetcher
        s = cfg["server"]
        self.server, self.camera = s["url"], s.get("camera") or "main"
        self.interval, self.timeout = float(s.get("interval_s", 1.0)), float(s.get("timeout_s", 5.0))
        c = cfg["collect"]
        self.same_diff, self.keep_every = float(c["same_diff"]), int(c["keep_every_same"])
        name = session or time.strftime("%Y%m%d_%H%M%S") + "_" + self.camera
        self.dir = Path(frames_root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.saved = len(list(self.dir.glob("f_*.jpg")))
        self.fails = self.skipped = 0
        self.last_stamp: Optional[str] = None
        self.last_small: Optional[np.ndarray] = None
        self.same_run = 0
        self.last_error = ""
        self.destroyed: Optional[str] = None
        self.stopped_reason = ""

    def step(self) -> Optional[Path]:
        """한 번 받아 보고, 저장했으면 경로를 돌려준다."""
        try:
            body, hdr = self.fetcher(self.server, self.code, self.camera, self.timeout)
        except FetchError as e:
            self.fails += 1
            self.last_error = str(e)
            if "403" in self.last_error:
                self.stopped_reason = "403 (드론 사진을 더 받을 수 없음)"
            return None
        self.destroyed = hdr.get("X-Destroyed")
        stamp = hdr.get("X-Captured-At")
        if stamp is not None and stamp == self.last_stamp:
            return None
        self.last_stamp = stamp
        small = _small_gray(body)
        if small is None:
            self.fails += 1
            self.last_error = "사진 해석 실패"
            return None
        if self.last_small is not None and float(np.mean(np.abs(small - self.last_small))) < self.same_diff:
            self.same_run += 1
            if self.same_run % max(1, self.keep_every) != 0:
                self.skipped += 1
                return None
        else:
            self.same_run = 0
        self.last_small = small
        self.saved += 1
        path = self.dir / f"f_{self.saved:05d}.jpg"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)  # 라벨 쪽이 쓰다 만 파일을 읽지 않게
        log_path = self.dir / "frames.csv"
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "received_at", "bytes", *HEADERS])
            w.writerow([path.name, f"{time.time():.3f}", len(body), *[hdr.get(k) or "" for k in HEADERS]])
        return path

    def run(self, stop: threading.Event, count: int = 0) -> None:
        log.info("수집 시작: %s (카메라 %s, %.1f초 간격)", self.dir, self.camera, self.interval)
        start_saved = self.saved
        while not stop.is_set() and (count <= 0 or self.saved - start_saved < count):
            t = time.time()
            self.step()
            if self.stopped_reason:
                log.warning("수집 멈춤: %s", self.stopped_reason)
                break
            if self.fails and self.fails % 10 == 0 and self.saved == start_saved:
                log.warning("받기 실패 %d번: %s — 서버 주소·임무코드·카메라 확인", self.fails, self.last_error)
            stop.wait(max(0.0, self.interval - (time.time() - t)))
        log.info("수집 끝: %d장 저장, 같은 사진 건너뜀 %d, 실패 %d", self.saved, self.skipped, self.fails)
