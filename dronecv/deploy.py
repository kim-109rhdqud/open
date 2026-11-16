"""모델을 로봇에 복사하고 로봇에서 점검(robot_check.py)을 돌린다.

로봇 폴더(~/m6_model):  best.onnx, model.json, robot_check.py, sample.jpg  (+ 이전 것 best_prev.onnx)
복사는 incoming/ 에 먼저 하고 다 받은 뒤 바꿔 끼운다(쓰다 만 모델을 읽지 않게).
ssh 는 비밀번호를 묻지 않는 키 방식이 필요하다 → `python run.py setup-key` 한 번.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT

log = logging.getLogger("deploy")
FILES = ("best.onnx", "model.json", "sample.jpg")


def _ssh_base(d: Dict) -> List[str]:
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new"]


def _remote_dir(d: Dict) -> str:
    r = str(d.get("remote_dir") or "m6_model")
    return r[2:] if r.startswith("~/") else r  # 홈 기준 상대 경로(scp 의 ~ 처리 차이를 피함)


def _run(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _last_json(text: str) -> Dict:
    for line in reversed((text or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"ok": False, "error": "점검 결과 JSON 없음"}


def deploy(work: Path, cfg: Dict, version: Optional[str] = None) -> Dict:
    d = cfg["deploy"]
    mode = d.get("mode", "ssh")
    cur_path = work / "models" / "current.json"
    if version is None:
        if not cur_path.exists():
            return {"ok": False, "error": "배포할 모델 없음"}
        version = json.loads(cur_path.read_text(encoding="utf-8"))["version"]
    src = work / "models" / version
    files = [src / f for f in FILES if (src / f).exists()] + [ROOT / "robot_check.py"]
    if not (src / "best.onnx").exists():
        return {"ok": False, "error": f"{version}/best.onnx 없음"}
    t0 = time.time()
    if mode == "off":
        return {"ok": False, "error": "deploy.mode=off"}
    if mode == "local":
        dst = Path(d.get("local_dir") or (work / "robot_sim"))
        inc = dst / "incoming"
        inc.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, inc / f.name)
        if (dst / "best.onnx").exists():
            shutil.copy2(dst / "best.onnx", dst / "best_prev.onnx")
        for f in inc.iterdir():
            os.replace(f, dst / f.name)
        p = _run([sys.executable, str(dst / "robot_check.py"), "--model", str(dst / "best.onnx"),
                  "--image", str(dst / "sample.jpg"), "--json"], 120)
        res = _last_json(p.stdout)
    else:
        host, user, port = d.get("host"), d.get("user"), str(d.get("port", 22))
        if not host or not user:
            return {"ok": False, "error": "deploy.host/user 설정 필요(local.yaml)"}
        target = f"{user}@{host}"
        rd = _remote_dir(d)
        q = shlex.quote
        p = _run(["ssh", "-p", port, *_ssh_base(d), target, f"mkdir -p {q(rd)}/incoming"], 30)
        if p.returncode != 0:
            return {"ok": False, "error": f"ssh 실패: {p.stderr.strip()[:200]}"}
        p = _run(["scp", "-P", port, *_ssh_base(d), *[str(f) for f in files], f"{target}:{rd}/incoming/"], 300)
        if p.returncode != 0:
            return {"ok": False, "error": f"scp 실패: {p.stderr.strip()[:200]}"}
        swap = (f"cd {q(rd)} && if [ -f best.onnx ]; then cp best.onnx best_prev.onnx; fi && "
                f"mv incoming/* . && {d.get('python', 'python3')} robot_check.py --model best.onnx --image sample.jpg --json")
        p = _run(["ssh", "-p", port, *_ssh_base(d), target, swap], 180)
        res = _last_json(p.stdout)
        if p.returncode != 0 and res.get("ok"):
            res["ok"] = False
        if not res.get("ok") and p.stderr:
            res.setdefault("error", p.stderr.strip()[:200])
    res.update(version=version, seconds=round(time.time() - t0, 1), mode=mode)
    (work / "models" / "deployed.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if res.get("ok"):
        log.info("배포 %s 완료: 로봇 OpenCV %s, 추론 %s ms, UGV %s대", version, res.get("opencv"), res.get("ms"), res.get("n"))
    else:
        log.warning("배포 %s 실패: %s", version, res.get("error"))
    return res


def setup_key(cfg: Dict) -> int:
    """키를 만들고 로봇에 등록(로봇 비밀번호를 한 번 묻는다)."""
    d = cfg["deploy"]
    host, user, port = d.get("host"), d.get("user"), str(d.get("port", 22))
    if not host or not user:
        print("local.yaml 에 deploy.host, deploy.user 를 먼저 적는다")
        return 2
    key = Path.home() / ".ssh" / "id_ed25519"
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    pub = key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    print(f"{user}@{host} 비밀번호를 한 번 입력한다.")
    p = subprocess.run(["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new", f"{user}@{host}",
                        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"],
                       input=pub + "\n", text=True)
    if p.returncode != 0:
        print("등록 실패")
        return 1
    ok = _run(["ssh", "-p", port, *_ssh_base(d), f"{user}@{host}", "echo ok"], 20)
    print("확인:", "성공 (이제 비밀번호 없이 배포됨)" if ok.stdout.strip() == "ok" else f"실패 {ok.stderr.strip()[:200]}")
    return 0 if ok.stdout.strip() == "ok" else 1
