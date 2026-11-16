"""학습용 목록 만들기: train.txt / val.txt / data.yaml.

시험(val)은 실사진만, 그리고 연속 사진 묶음 단위로 뗀다(바로 옆 사진이 학습에 있으면 점수가 부풀려짐).
심볼릭 링크 대신 경로 목록 파일을 써서 Windows 에서도 관리자 권한 없이 된다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml


def build(work: Path, cfg: Dict) -> Tuple[Path, int, int]:
    d = cfg["dataset"]
    block, every = int(d["block"]), int(d["val_every"])
    real = sorted((work / "dataset" / "real" / "images").glob("*.jpg"))
    synth = sorted((work / "dataset" / "synth" / "images").glob("*.jpg"))
    val = [p for i, p in enumerate(real) if (i // block) % every == every - 1]
    if not val and real:
        val = real[-max(1, len(real) // 5):]
    vset = set(val)
    real_train = [p for p in real if p not in vset]
    cap = int(d.get("max_real_train", 0))
    if cap >= 2 and len(real_train) > cap:
        idx = sorted(set(round(i * (len(real_train) - 1) / (cap - 1)) for i in range(cap)))
        real_train = [real_train[i] for i in idx]
    train = real_train + synth
    ds = work / "dataset"
    (ds / "train.txt").write_text("\n".join(str(p.resolve()) for p in train) + "\n", encoding="utf-8")
    (ds / "val.txt").write_text("\n".join(str(p.resolve()) for p in val) + "\n", encoding="utf-8")
    data = {"path": str(ds.resolve()), "train": str((ds / "train.txt").resolve()),
            "val": str((ds / "val.txt").resolve()), "names": {0: "ugv"}}
    yml = ds / "data.yaml"
    yml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml, len(train), len(val)
