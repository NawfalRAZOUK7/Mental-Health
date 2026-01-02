#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = os.getenv("MHP_VERSION", "v1").strip() or "v1"

DATA_RAW = REPO_ROOT / "data_raw"
VERSION_DIR = REPO_ROOT / VERSION
DATA_CLEAN = VERSION_DIR / "data_clean"
REPORT_DIR = VERSION_DIR / "report"
ASSETS_DIR = REPO_ROOT / "assets"


def ensure_dirs() -> tuple[Path, Path]:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CLEAN, REPORT_DIR
