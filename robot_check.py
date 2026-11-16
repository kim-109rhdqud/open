#!/usr/bin/env python3
"""드론 사진 UGV 검출 (ONNX + OpenCV DNN). 로봇에서 그대로 돈다: numpy·OpenCV 만 필요, PyTorch 불필요.

점검:   python3 robot_check.py --model best.onnx --image sample.jpg [--bench 10] [--json]
코드에서: from robot_check import load, detect, team_of
          net, meta = load("best.onnx");  dets = detect(net, img, meta)

model.json(같은 폴더)에 입력 크기(imgsz)·클래스 이름이 있다. 없으면 640.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def load(model_path):
    net = cv2.dnn.readNetFromONNX(model_path)
    meta = {"imgsz": 640, "names": ["ugv"], "conf": 0.25, "iou": 0.5}
    mj = os.path.join(os.path.dirname(os.path.abspath(model_path)), "model.json")
    if os.path.exists(mj):
        with open(mj, encoding="utf-8") as f:
            meta.update(json.load(f))
    return net, meta


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, dx, dy


def detect(net, img, meta, conf=None, iou=None):
    """→ [{'box': [x1,y1,x2,y2], 'conf': float, 'cls': int, 'team': 'RED'|'BLUE'|None}] (원본 픽셀)"""
    size = int(meta["imgsz"])
    conf = float(meta.get("conf", 0.25) if conf is None else conf)
    iou = float(meta.get("iou", 0.5) if iou is None else iou)
    lb, r, dx, dy = letterbox(img, size)
    net.setInput(cv2.dnn.blobFromImage(lb, 1 / 255.0, (size, size), swapRB=True, crop=False))
    out = net.forward()
    pred = out[0] if out.ndim == 3 else out  # (4+nc, N)
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T
    scores = pred[4:].max(axis=0)
    cls = pred[4:].argmax(axis=0)
    keep = scores >= conf
    if not keep.any():
        return []
    cx, cy, w, h = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
    sc, cl = scores[keep], cls[keep]
    x1 = (cx - w / 2 - dx) / r
    y1 = (cy - h / 2 - dy) / r
    bw, bh = w / r, h / r
    rects = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
    idx = cv2.dnn.NMSBoxes(rects, [float(s) for s in sc], conf, iou)
    H, W = img.shape[:2]
    dets = []
    for i in np.array(idx).reshape(-1):
        x, y, ww, hh = rects[i]
        box = [max(0, int(x)), max(0, int(y)), min(W, int(x + ww)), min(H, int(y + hh))]
        dets.append({"box": box, "conf": round(float(sc[i]), 3), "cls": int(cl[i]),
                     "team": team_of(img[box[1]:box[3], box[0]:box[2]])})
    return dets


def team_of(crop, s_min=110, v_min=70, min_pixels=6):
    """안테나 색으로 팀: 'RED' | 'BLUE' | None."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vivid = (s >= s_min) & (v >= v_min)
    red = int((vivid & ((h <= 9) | (h >= 170))).sum())
    blue = int((vivid & (h >= 100) & (h <= 130)).sum())
    if max(red, blue) < min_pixels or max(red, blue) < 2 * min(red, blue):
        return None
    return "RED" if red > blue else "BLUE"


def main(argv=None):
    ap = argparse.ArgumentParser(description="드론 UGV 검출 점검")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--bench", type=int, default=3, help="반복 횟수(시간 재기)")
    ap.add_argument("--json", action="store_true", help="마지막 줄에 JSON 한 줄")
    ap.add_argument("--out", default="", help="상자 그린 그림 저장")
    a = ap.parse_args(argv)
    res = {"ok": False, "opencv": cv2.__version__, "python": sys.version.split()[0]}
    try:
        net, meta = load(a.model)
        img = cv2.imread(a.image)
        if img is None:
            raise RuntimeError("사진을 못 읽음")
        dets, times = [], []
        for _ in range(max(1, a.bench)):
            t = time.time()
            dets = detect(net, img, meta)
            times.append((time.time() - t) * 1000)
        res.update(ok=True, imgsz=meta["imgsz"], version=meta.get("version"), ms=round(float(np.median(times)), 1),
                   n=len(dets), detections=dets)
        if a.out:
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(a.out, img)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    if not a.json:
        print(f"OpenCV {res['opencv']}, Python {res['python']}")
        if res["ok"]:
            print(f"모델 {res.get('version')} 입력 {res['imgsz']}  추론 {res['ms']} ms  UGV {res['n']}대")
            for d in res["detections"]:
                print(f"  {d['box']} conf={d['conf']} team={d['team']}")
        else:
            print("실패:", res.get("error"))
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
