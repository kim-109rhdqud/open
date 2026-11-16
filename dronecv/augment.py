"""잘라 붙이기 증강: 우리 로봇 한 대의 실사진에서 여러 대가 있는 사진을 만든다.

재료: work/dataset/crops/*.png (로봇만 남긴 투명 PNG, 자동 라벨이 만듦), work/backgrounds/*.png (빈 경기장).
만드는 것: work/dataset/synth/images, labels. 매번 지우고 새로 만든다.
- 로봇을 무작위 위치·각도(0~360°)·크기(±10%)·밝기로 2~6대 붙임
- 안테나 색을 빨강↔파랑으로 바꿔 상대 팀 모습도 만듦
- UGV 없는 배경 사진도 조금 섞음(잘못 찾기 줄이기)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import vision

log = logging.getLogger("augment")


def swap_team_color(bgra: np.ndarray) -> np.ndarray:
    """선명한 빨강 ↔ 파랑을 맞바꾼다(채도·밝기는 그대로)."""
    bgr = bgra[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1], hsv[:, :, 2]
    vivid = (s >= 90) & (v >= 60)
    red = vivid & ((h <= 9) | (h >= 170))
    blue = vivid & (h >= 100) & (h <= 130)
    h2 = h.copy()
    h2[red] = 115
    h2[blue] = 0
    hsv[:, :, 0] = h2.astype(np.uint8)
    out = bgra.copy()
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def rotate_bgra(bgra: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = bgra.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(bgra, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def paste(dst: np.ndarray, bgra: np.ndarray, x: int, y: int) -> Optional[Tuple[int, int, int, int]]:
    """dst 에 (x,y) 왼쪽 위로 붙이고, 실제로 보이는 부분의 상자를 돌려준다."""
    H, W = dst.shape[:2]
    h, w = bgra.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    src = bgra[y1 - y:y2 - y, x1 - x:x2 - x]
    a = (src[:, :, 3:4].astype(np.float32) / 255.0)
    roi = dst[y1:y2, x1:x2].astype(np.float32)
    dst[y1:y2, x1:x2] = (roi * (1 - a) + src[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    ys, xs = np.nonzero(src[:, :, 3] > 127)
    if len(xs) < 10:
        return None
    return x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1


def synthesize(work: Path, cfg: Dict, seed: Optional[int] = None) -> int:
    acfg = cfg["augment"]
    rng = np.random.default_rng(seed)
    crops = sorted((work / "dataset" / "crops").glob("*.png"))
    bgs = sorted((work / "backgrounds").glob("*.png"))
    n_real = len(list((work / "dataset" / "real" / "labels").glob("*.txt")))
    out = work / "dataset" / "synth"
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    if not crops or not bgs:
        log.info("합성 건너뜀 (잘라 낸 로봇 %d, 배경 %d)", len(crops), len(bgs))
        return 0
    total = min(int(acfg["max_total"]), int(acfg["per_real"]) * max(1, n_real))
    crop_imgs = [vision_read_bgra(p) for p in crops]
    crop_imgs = [c for c in crop_imgs if c is not None]
    bg_imgs = [vision.read(p) for p in bgs]
    bg_imgs = [b for b in bg_imgs if b is not None]
    made = 0
    for i in range(total):
        bg = bg_imgs[int(rng.integers(len(bg_imgs)))].copy()
        H, W = bg.shape[:2]
        lines: List[str] = []
        if rng.random() >= float(acfg["negatives"]):
            k = int(rng.integers(int(acfg["robots_min"]), int(acfg["robots_max"]) + 1))
            boxes: List[Tuple[int, int, int, int]] = []
            for _ in range(k * 4):  # 겹침이 심하면 다시 뽑음
                if len(boxes) >= k:
                    break
                c = crop_imgs[int(rng.integers(len(crop_imgs)))]
                if rng.random() < float(acfg["swap_color_p"]):
                    c = swap_team_color(c)
                c = rotate_bgra(c, float(rng.uniform(0, 360)), float(rng.uniform(0.9, 1.1)))
                gain = float(rng.uniform(0.85, 1.15))
                c[:, :, :3] = np.clip(c[:, :, :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
                h, w = c.shape[:2]
                if w >= W or h >= H:
                    continue
                x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
                cand = (x, y, x + w, y + h)
                if any(vision.iou(cand, b) > 0.3 for b in boxes):
                    continue
                b = paste(bg, c, x, y)
                if b is None:
                    continue
                boxes.append(b)
            for (x1, y1, x2, y2) in boxes:
                lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        name = f"synth_{i:05d}"
        vision.write(out / "images" / f"{name}.jpg", bg)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        made += 1
    log.info("합성 사진 %d장 (로봇 조각 %d, 배경 %d)", made, len(crop_imgs), len(bg_imgs))
    return made


def vision_read_bgra(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    return img if img is not None and img.ndim == 3 and img.shape[2] == 4 else None
