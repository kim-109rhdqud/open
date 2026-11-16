"""설정 읽기: config.yaml 위에 local.yaml 을 덮어쓴다. 비밀(임무코드·API 키)은 환경변수나 숨김 입력으로만."""
from __future__ import annotations

import copy
import getpass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    local = Path(path) if path else ROOT / "local.yaml"
    if local.exists():
        base = _merge(base, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    if overrides:
        base = _merge(base, overrides)
    if not base["server"].get("url"):
        base["server"]["url"] = (os.environ.get("MAICON_SERVER_URL") or os.environ.get("MAICON_SERVER")
                                 or os.environ.get("MAICON_BASE") or "")
    return base


def work_dir(cfg: Dict[str, Any]) -> Path:
    p = Path(cfg["work_dir"])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def mission_code(ask: bool = True) -> str:
    code = os.environ.get("MAICON_CODE") or os.environ.get("MAICON_MISSION_CODE") or ""
    if not code and ask:
        code = getpass.getpass("임무코드(화면에 안 보임): ").strip()
    return code


def claude_key(ask: bool = True) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and ask:
        key = getpass.getpass("Claude API 키(없으면 Enter → OpenCV 만으로 라벨): ").strip()
    return key
