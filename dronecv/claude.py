"""Claude Vision 으로 OpenCV 후보 검증 (준비·연습 때 라벨 만들기 전용. 경기 중 로봇에서는 쓰지 않는다).

Claude 에게 상자 좌표를 새로 그리게 하지 않는다(좌표가 부정확). OpenCV 가 그린 번호 상자마다
'UGV 인가 / 팀 색 / 로봇 전체를 감싸는가'를 묻고, 상자에 안 들어간 UGV 가 있는지 센다.
API 키는 헤더로만 보내고 어디에도 기록하지 않는다.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests

from .vision import Candidate, crop_tile, draw

API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = """경기장을 위에서 고정 카메라로 찍은 사진이다.
UGV는 작은 6륜 로봇(약 25×23 cm)으로, 빨강 또는 파랑 안테나 띠와 흰 수광부가 있다.
검정·회색 정육면체(40 cm)는 상자이고 UGV가 아니다. 사람·손·그림자·바닥 무늬도 UGV가 아니다.

첫 번째 그림: 전체 사진 위에 노란 번호 상자(후보)를 그렸다.
두 번째 그림: 각 후보를 확대한 것(번호 같음).

번호마다 판단해서 report 도구로 답하라.
- is_ugv: 그 상자 안의 주인공이 UGV 인가
- team: 안테나 색 RED / BLUE / UNKNOWN
- tight: 상자가 UGV 전체를 대체로 감싸고(잘린 곳 없이) 다른 UGV를 함께 담지 않았는가
- missed: 어떤 번호 상자에도 들어가지 않은 UGV 가 사진에 몇 대 있는가
확실하지 않으면 is_ugv=false 로 답한다."""

TOOL = {
    "name": "report",
    "description": "후보 상자별 판정",
    "input_schema": {
        "type": "object",
        "properties": {
            "boxes": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_ugv": {"type": "boolean"},
                    "team": {"type": "string", "enum": ["RED", "BLUE", "UNKNOWN"]},
                    "tight": {"type": "boolean"},
                },
                "required": ["id", "is_ugv", "team", "tight"]}},
            "missed": {"type": "integer"},
            "note": {"type": "string"},
        },
        "required": ["boxes", "missed"],
    },
}


def _jpeg_b64(img: np.ndarray, max_side: int) -> str:
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


class ClaudeVerifier:
    def __init__(self, api_key: str, model: str, timeout: float = 60.0, max_side: int = 1568,
                 post: Callable = requests.post, retries: int = 2) -> None:
        self._key = api_key
        self.model, self.timeout, self.max_side = model, timeout, max_side
        self.post, self.retries = post, retries
        self.calls = self.errors = 0

    def __repr__(self) -> str:  # 키가 로그에 찍히지 않게
        return f"ClaudeVerifier(model={self.model!r})"

    def verify(self, img: np.ndarray, cands: Sequence[Candidate]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": _jpeg_b64(draw(img, cands), self.max_side)}}]
        tile = crop_tile(img, cands)
        if tile is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": _jpeg_b64(tile, self.max_side)}})
        content.append({"type": "text", "text": PROMPT + f"\n후보 수: {len(cands)}"})
        body = {"model": self.model, "max_tokens": 1024, "tools": [TOOL],
                "tool_choice": {"type": "tool", "name": "report"},
                "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        last = ""
        for attempt in range(self.retries + 1):
            self.calls += 1
            try:
                r = self.post(API_URL, headers=headers, data=json.dumps(body), timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 529):
                    last = f"HTTP {r.status_code}"
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                for block in r.json().get("content", []):
                    if block.get("type") == "tool_use" and block.get("name") == "report":
                        return _clean(block.get("input") or {}, len(cands))
                raise RuntimeError("도구 응답 없음")
            except requests.RequestException as e:
                last = type(e).__name__
                time.sleep(2 * (attempt + 1))
        self.errors += 1
        raise RuntimeError(f"Claude 실패: {last}")


def _clean(res: Dict[str, Any], n: int) -> Dict[str, Any]:
    boxes = {}
    for b in res.get("boxes") or []:
        try:
            i = int(b["id"])
        except Exception:
            continue
        if 1 <= i <= n:
            boxes[i] = {"is_ugv": bool(b.get("is_ugv")), "team": str(b.get("team", "UNKNOWN")).upper(),
                        "tight": bool(b.get("tight"))}
    try:
        missed = max(0, int(res.get("missed", 0)))
    except Exception:
        missed = 99
    return {"boxes": boxes, "missed": missed, "note": str(res.get("note", ""))[:200]}
