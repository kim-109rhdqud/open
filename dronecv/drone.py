"""경기 서버에서 드론 사진 한 장 받기 (팀 레포 teleop_rosbag/uav_collect.py 와 같은 요청)."""
from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

HEADERS = ("X-Camera", "X-Captured-At", "X-Frame-Interval-S", "X-Test-Mode", "X-Destroyed", "X-Camp")


class FetchError(RuntimeError):
    """주소·임무코드가 담기지 않은 짧은 이유만 가진다."""


def fetch(server: str, code: str, camera: Optional[str], timeout: float) -> Tuple[bytes, Dict[str, Optional[str]]]:
    params = {"mission_code": code}
    if camera:
        params["camera"] = camera
    url = f"{server.rstrip('/')}/api/report/drone?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
            body, hdr = resp.read(), resp.headers
    except Exception as e:  # 메시지에 주소·코드가 섞이지 않게 형식만
        raise FetchError(type(e).__name__ + (f" {e.code}" if hasattr(e, "code") else "")) from None
    ctype = str(hdr.get("Content-Type", ""))
    if not ctype.lower().startswith("image/jpeg") or not body:
        raise FetchError(f"JPEG 아님 (Content-Type={ctype[:40]!r}, {len(body)} bytes)")
    return body, {k: hdr.get(k) for k in HEADERS}
