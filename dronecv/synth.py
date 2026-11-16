"""시험용 가짜 드론 사진(실물과 다르다 — 코드가 끝까지 도는지만 본다).

밝은 바닥 + 격자선 + 검정 상자 + UGV(어두운 몸체, 빨강/파랑 안테나, 흰 수광부)가 돌아다닌다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Bot:
    x: float
    y: float
    th: float
    team: str = "BLUE"
    v: float = 9.0


class Arena:
    def __init__(self, w: int = 960, h: int = 720, n_bots: int = 1, seed: int = 0, boxes: int = 8) -> None:
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed)
        base = np.full((h, w, 3), (165, 170, 172), np.uint8)
        for x in range(0, w, 60):
            cv2.line(base, (x, 0), (x, h), (150, 155, 158), 1)
        for y in range(0, h, 60):
            cv2.line(base, (0, y), (w, y), (150, 155, 158), 1)
        for _ in range(boxes):
            bx, by = int(self.rng.integers(40, w - 80)), int(self.rng.integers(40, h - 80))
            cv2.rectangle(base, (bx, by), (bx + 40, by + 40), (35, 35, 38), -1)
        self.base = base
        teams = ["BLUE", "RED"]
        self.bots = [Bot(float(self.rng.uniform(100, w - 100)), float(self.rng.uniform(100, h - 100)),
                         float(self.rng.uniform(0, 2 * math.pi)), teams[i % 2]) for i in range(n_bots)]

    def step(self) -> None:
        for b in self.bots:
            b.th += float(self.rng.normal(0, 0.35))
            nx, ny = b.x + b.v * math.cos(b.th), b.y + b.v * math.sin(b.th)
            if not (40 < nx < self.w - 40 and 40 < ny < self.h - 40):
                b.th += math.pi
                continue
            b.x, b.y = nx, ny

    def render(self) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, str]]]:
        img = self.base.copy()
        truth = []
        for b in self.bots:
            rect = ((b.x, b.y), (30.0, 26.0), math.degrees(b.th))
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(img, [pts], (60, 62, 66))
            c, s = math.cos(b.th), math.sin(b.th)
            ax, ay = int(b.x - 8 * c), int(b.y - 8 * s)
            cv2.circle(img, (ax, ay), 5, (40, 40, 220) if b.team == "RED" else (220, 90, 30), -1)
            cv2.circle(img, (int(b.x + 7 * c), int(b.y + 7 * s)), 3, (245, 245, 245), -1)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            truth.append((int(x1), int(y1), int(x2), int(y2), b.team))
        gain = float(self.rng.uniform(0.97, 1.03))
        noise = self.rng.normal(0, 2.0, img.shape)
        img = np.clip(img.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
        return img, truth

    def jpeg(self) -> Tuple[bytes, List]:
        img, truth = self.render()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes(), truth
