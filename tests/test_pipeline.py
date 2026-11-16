"""빠른 시험(학습 없이). 학습까지 포함한 전체 흐름은 `python run.py selftest`."""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dronecv import augment, config as C, dataset, vision  # noqa: E402
from dronecv.autolabel import AutoLabeler  # noqa: E402
from dronecv.claude import ClaudeVerifier  # noqa: E402
from dronecv.collect import Collector  # noqa: E402
from dronecv.drone import FetchError, fetch  # noqa: E402
from dronecv.fakeserver import FakeDroneServer  # noqa: E402
from dronecv.synth import Arena  # noqa: E402
import robot_check  # noqa: E402

CODE = "SECRET-CODE-123"


@pytest.fixture()
def cfg(tmp_path):
    c = C.load(overrides={"work_dir": str(tmp_path / "work"), "server": {"interval_s": 0.0}})
    return c


@pytest.fixture()
def server():
    s = FakeDroneServer(code=CODE).start()
    yield s
    s.stop()


def collect(cfg, server, work, n=60):
    cfg["server"]["url"] = server.url
    col = Collector(cfg, CODE, work / "frames", session="s1")
    col.run(threading.Event(), count=n)
    return col


def test_fetch_wrong_code_hides_secret(server):
    with pytest.raises(FetchError) as e:
        fetch(server.url, "WRONG", "main", 3)
    assert "401" in str(e.value) and "WRONG" not in str(e.value) and "127.0.0.1" not in str(e.value)


def test_collect_saves_and_never_writes_code(cfg, server, tmp_path):
    work = C.work_dir(cfg)
    col = collect(cfg, server, work, 20)
    files = sorted((work / "frames" / "s1").glob("f_*.jpg"))
    assert len(files) == 20 and col.fails == 0
    for p in (work / "frames" / "s1").iterdir():
        if p.suffix in (".csv", ".log", ".json"):
            assert CODE not in p.read_text(encoding="utf-8")


def test_collect_skips_same_picture(cfg, tmp_path):
    work = C.work_dir(cfg)
    a = Arena(seed=1)
    body, _ = a.jpeg()
    stamps = iter(range(1000))
    fake = lambda *args: (body, {"X-Captured-At": str(next(stamps)), "X-Camera": "main"})  # 같은 사진, 다른 시각
    cfg["server"]["url"] = "http://x"
    col = Collector(cfg, CODE, work / "frames", fetcher=fake, session="same")
    for _ in range(25):
        col.step()
    assert col.saved == 1 + 24 // cfg["collect"]["keep_every_same"]
    assert col.skipped > 15


def test_candidates_find_moving_robot():
    a = Arena(seed=3)
    frames = []
    for _ in range(30):
        a.step()
        frames.append(a.render())
    bg = vision.median_background([f[0] for f in frames])
    lcfg = C.load()["label"]
    found = 0
    for img, truth in frames:
        cs = vision.candidates(img, bg, lcfg)
        ugv = [c for c in cs if c.fg and c.team]
        if len(ugv) == 1 and vision.iou(ugv[0].box, truth[0][:4]) > 0.4 and ugv[0].team == truth[0][4]:
            found += 1
    assert found >= 27


def test_autolabel_opencv_only(cfg, server):
    work = C.work_dir(cfg)
    collect(cfg, server, work, 40)
    lab = AutoLabeler(cfg, work, None)
    new = lab.process([work / "frames"])
    assert new["ok"] >= 32
    lbls = list((work / "dataset" / "real" / "labels").glob("*.txt"))
    assert len(lbls) == new["ok"]
    line = lbls[0].read_text().split()
    assert line[0] == "0" and all(0 < float(v) < 1 for v in line[1:])
    assert list((work / "dataset" / "crops").glob("*.png"))
    assert lab.process([work / "frames"])["ok"] == 0  # 다시 돌려도 같은 사진은 안 함


class FakeClaude:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def verify(self, img, cands):
        self.calls += 1
        return self.answer(cands)


def test_autolabel_claude_rules(cfg, server):
    work = C.work_dir(cfg)
    collect(cfg, server, work, 15)
    yes = FakeClaude(lambda cs: {"boxes": {1: {"is_ugv": True, "team": "RED", "tight": True}}, "missed": 0})
    lab = AutoLabeler(cfg, work, yes)
    assert lab.process([work / "frames"])["ok"] >= 12
    assert yes.calls >= 12
    # 놓친 UGV 가 있다고 하면 버림
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["work_dir"] = str(Path(cfg["work_dir"]).parent / "w2")
    w2 = C.work_dir(cfg2)
    collect(cfg2, server, w2, 15)
    missed = FakeClaude(lambda cs: {"boxes": {1: {"is_ugv": True, "team": "RED", "tight": True}}, "missed": 1})
    lab2 = AutoLabeler(cfg2, w2, missed)
    r = lab2.process([w2 / "frames"])
    assert r["ok"] == 0 and r["rejected"] >= 12


def test_claude_request_and_key_not_leaked():
    seen = {}

    class R:
        status_code = 200

        def json(self):
            return {"content": [{"type": "tool_use", "name": "report",
                                 "input": {"boxes": [{"id": 1, "is_ugv": True, "team": "BLUE", "tight": True},
                                                     {"id": 9, "is_ugv": True, "team": "RED", "tight": True}],
                                           "missed": 0}}]}

    def post(url, headers, data, timeout):
        seen.update(url=url, headers=headers, body=json.loads(data))
        return R()

    v = ClaudeVerifier("sk-test-KEY", "claude-opus-5-5", post=post)
    a = Arena(seed=2)
    img, truth = a.render()
    res = v.verify(img, [vision.Candidate(truth[0][:4])])
    assert res["boxes"] == {1: {"is_ugv": True, "team": "BLUE", "tight": True}}  # 없는 번호(9)는 버림
    assert seen["headers"]["x-api-key"] == "sk-test-KEY"
    assert seen["body"]["tool_choice"]["name"] == "report"
    assert "sk-test-KEY" not in repr(v) and "sk-test-KEY" not in json.dumps(seen["body"])


def test_augment_and_dataset_split(cfg, server):
    work = C.work_dir(cfg)
    collect(cfg, server, work, 60)
    AutoLabeler(cfg, work, None).process([work / "frames"])
    n = augment.synthesize(work, cfg, seed=0)
    assert n > 0
    multi = 0
    for lbl in (work / "dataset" / "synth" / "labels").glob("*.txt"):
        rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()]
        for r in rows:
            assert r[0] == "0" and all(0 <= float(v) <= 1 for v in r[1:])
        multi += len(rows) >= 2
    assert multi > n * 0.5
    yml, n_train, n_val = dataset.build(work, cfg)
    val = (work / "dataset" / "val.txt").read_text().split()
    assert n_val > 0 and all("synth" not in v for v in val)
    train = set((work / "dataset" / "train.txt").read_text().split())
    assert not train & set(val)


def test_swap_team_color():
    crop = np.zeros((20, 20, 4), np.uint8)
    crop[:, :, 3] = 255
    crop[5:10, 5:10, :3] = (40, 40, 220)  # 빨강
    assert vision.team_of(crop[:, :, :3], C.load()["label"]["color"])[0] == "RED"
    sw = augment.swap_team_color(crop)
    assert vision.team_of(sw[:, :, :3], C.load()["label"]["color"])[0] == "BLUE"


def test_robot_check_team_and_letterbox():
    img = np.full((30, 30, 3), 150, np.uint8)
    img[10:16, 10:16] = (220, 90, 30)
    assert robot_check.team_of(img) == "BLUE"
    lb, r, dx, dy = robot_check.letterbox(np.zeros((480, 640, 3), np.uint8), 320)
    assert lb.shape == (320, 320, 3) and r == 0.5 and dx == 0 and dy == 40


def test_deploy_needs_model(cfg):
    from dronecv.deploy import deploy
    assert deploy(C.work_dir(cfg), cfg)["ok"] is False


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="학습 포함, RUN_SLOW=1 일 때만")
def test_selftest_end_to_end():
    import run
    assert run.main(["selftest", "--frames", "80", "--epochs", "5"]) == 0


def test_limits_per_cycle_and_real_cap(cfg, server):
    cfg["label"]["max_per_cycle"] = 10
    cfg["dataset"]["max_real_train"] = 5
    work = C.work_dir(cfg)
    collect(cfg, server, work, 40)
    lab = AutoLabeler(cfg, work, None)
    r = lab.process([work / "frames"])
    assert r["ok"] + r["rejected"] == 10            # 한 주기에 10장만
    while sum(lab.process([work / "frames"]).values()):
        pass
    assert lab.counts()["total"] == 40              # 다음 주기들에 나머지
    yml, n_train, n_val = dataset.build(work, cfg)
    assert n_train == 5 and n_val > 0               # 합성 없음, 실사진 5장으로 솎음
