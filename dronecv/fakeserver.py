"""시험용 가짜 경기 서버: GET /api/report/drone 만 흉내 낸다(요청마다 로봇이 한 걸음 움직인 새 사진)."""
from __future__ import annotations

import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from .synth import Arena


class FakeDroneServer:
    def __init__(self, code: str = "TEST-CODE", n_bots: int = 1, seed: int = 0, still_every: int = 0) -> None:
        self.code, self.arena, self.n = code, Arena(n_bots=n_bots, seed=seed), 0
        self.still_every = still_every
        self.lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 조용히
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                if u.path != "/api/report/drone":
                    self.send_response(404); self.end_headers(); return
                if q.get("mission_code", [""])[0] != outer.code:
                    self.send_response(401); self.end_headers(); return
                with outer.lock:
                    outer.n += 1
                    if not (outer.still_every and outer.n % outer.still_every == 0):
                        outer.arena.step()
                    body, _ = outer.arena.jpeg()
                    stamp = f"{time.time():.3f}"
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Camera", q.get("camera", ["main"])[0])
                self.send_header("X-Captured-At", stamp)
                self.send_header("X-Frame-Interval-S", "1")
                self.send_header("X-Destroyed", "0")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def start(self) -> "FakeDroneServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
