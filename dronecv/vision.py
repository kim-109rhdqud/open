"""OpenCV 후보 찾기: 배경(중앙값) 차분으로 움직이는 덩어리 + 안테나 색. 학습 없이 동작한다.

고정 카메라라 여러 장의 중앙값이 곧 '빈 경기장'이 된다. 로봇이 돌아다니면 중앙값에서 빠지므로
현재 사진과 배경의 차이가 곧 로봇이다. 멈춰 있는 로봇은 안테나 색 덩어리로 보충한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (픽셀)


@dataclass
class Candidate:
    box: Box
    team: Optional[str] = None      # RED | BLUE | None
    fg: bool = False                # 배경 차분에서 나온 덩어리인가
    color_px: Dict[str, int] = field(default_factory=dict)
    mask: Optional[np.ndarray] = None   # box 크기의 0/255 전경 마스크(잘라 붙이기용)


def read(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)  # 한글 경로(Windows)에서도 읽히게
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write(path: Path, img: np.ndarray, ext: str = ".jpg") -> None:
    ok, buf = cv2.imencode(ext, img)
    if ok:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))


def median_background(images: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack([im for im in images if im is not None], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def color_masks(img: np.ndarray, ccfg: Dict) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vivid = (s >= int(ccfg["s_min"])) & (v >= int(ccfg["v_min"]))
    return {"RED": vivid & ((h <= 9) | (h >= 170)), "BLUE": vivid & (h >= 100) & (h <= 130)}


def team_of(crop: np.ndarray, ccfg: Dict) -> Tuple[Optional[str], Dict[str, int]]:
    if crop is None or crop.size == 0:
        return None, {}
    m = color_masks(crop, ccfg)
    n = {k: int(v.sum()) for k, v in m.items()}
    best = max(n, key=n.get)
    other = min(n, key=n.get)
    if n[best] >= int(ccfg["min_pixels"]) and n[best] >= 2 * n[other]:
        return best, n
    return None, n


def _pad(box: Box, pad: float, shape) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = int(round(w * pad)), int(round(h * pad))
    H, W = shape[:2]
    return max(0, x1 - dx), max(0, y1 - dy), min(W, x2 + dx), min(H, y2 + dy)


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def foreground(img: np.ndarray, bg: np.ndarray, thr: int) -> np.ndarray:
    diff = cv2.absdiff(img, bg).max(axis=2)
    mask = (diff >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def candidates(img: np.ndarray, bg: Optional[np.ndarray], lcfg: Dict,
               area_range: Optional[Tuple[float, float]] = None,
               size_hint: Optional[Tuple[int, int]] = None) -> List[Candidate]:
    ccfg = lcfg["color"]
    amin, amax = area_range or (float(lcfg["min_area_px"]), float(lcfg["max_area_px"]))
    out: List[Candidate] = []
    if bg is not None and bg.shape == img.shape:
        mask = foreground(img, bg, int(lcfg["fg_threshold"]))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (amin <= area <= amax):
                continue
            box = _pad((int(x), int(y), int(x + w), int(y + h)), float(lcfg["pad"]), img.shape)
            x1, y1, x2, y2 = box
            team, px = team_of(img[y1:y2, x1:x2], ccfg)
            out.append(Candidate(box, team, True, px, mask[y1:y2, x1:x2].copy()))
    # 멈춰 있는 로봇: 배경 덩어리에 안 걸린 안테나 색 → 크기 힌트로 상자를 만든다
    if size_hint is not None:
        sw, sh = size_hint
        for team, m in color_masks(img, ccfg).items():
            n, _, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8) * 255, 8)
            for i in range(1, n):
                if stats[i][4] < int(ccfg["min_pixels"]):
                    continue
                cx, cy = cents[i]
                box = (int(cx - sw / 2), int(cy - sh / 2), int(cx + sw / 2), int(cy + sh / 2))
                H, W = img.shape[:2]
                box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
                if any(iou(box, c.box) > 0.1 or _inside(cx, cy, c.box) for c in out):
                    continue
                out.append(Candidate(box, team, False, {team: int(stats[i][4])}, None))
    return _nms(out)


def _inside(x: float, y: float, b: Box) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def _nms(cs: List[Candidate], thr: float = 0.5) -> List[Candidate]:
    cs = sorted(cs, key=lambda c: (c.fg, (c.box[2] - c.box[0]) * (c.box[3] - c.box[1])), reverse=True)
    keep: List[Candidate] = []
    for c in cs:
        if all(iou(c.box, k.box) < thr for k in keep):
            keep.append(c)
    return keep


def draw(img: np.ndarray, cs: Sequence[Candidate], numbered: bool = True,
         colors: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
    out = img.copy()
    thick = max(1, img.shape[1] // 800)
    for i, c in enumerate(cs, 1):
        col = colors[i - 1] if colors else (0, 255, 255)
        x1, y1, x2, y2 = c.box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        if numbered:
            cv2.putText(out, str(i), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * thick + 0.2, col, thick, cv2.LINE_AA)
    return out


def crop_tile(img: np.ndarray, cs: Sequence[Candidate], size: int = 160) -> Optional[np.ndarray]:
    tiles = []
    for i, c in enumerate(cs, 1):
        x1, y1, x2, y2 = _pad(c.box, 0.5, img.shape)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        s = size / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_CUBIC)
        t = np.full((size + 24, size, 3), 40, np.uint8)
        t[24:24 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(t, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return None
    cols = min(6, len(tiles))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    blank = np.full_like(tiles[0], 40)
    return np.vstack([np.hstack(r + [blank] * (cols - len(r))) for r in rows])
