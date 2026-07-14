"""Env loading — stdlib only, keeps secrets out of the code.

Moved verbatim from scripts/nebius_macro_validate.py so every entrypoint (scripts,
jobs, serving) shares one loader. Shell env always wins over .env.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Minimal stdlib .env loader (shell env wins). Keeps the key out of the code."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def env(*names: str, default: str = "") -> str:
    """First set env var among `names`, else `default`."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# Back-compat aliases: scripts/ imported these underscored names.
_load_dotenv = load_dotenv
_env = env
