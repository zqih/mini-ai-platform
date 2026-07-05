from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def storage_root() -> Path:
    default = project_root() / "storage"
    return Path(os.environ.get("MINI_AI_PLATFORM_STORAGE", default)).resolve()


def db_path() -> Path:
    default = storage_root() / "mini_ai_platform.db"
    return Path(os.environ.get("MINI_AI_PLATFORM_DB", default)).resolve()


def ensure_runtime_dirs() -> None:
    root = storage_root()
    for child in ["artifacts", "datasets", "logs", "models"]:
        (root / child).mkdir(parents=True, exist_ok=True)
    db_path().parent.mkdir(parents=True, exist_ok=True)

