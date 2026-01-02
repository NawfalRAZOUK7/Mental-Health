#!/usr/bin/env python3
"""Run the full v1 pipeline in order."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STEPS = [
    "src/01_country_mapping.py",
    "src/02_clean_who.py",
    "src/03_clean_gbd.py",
    "src/04_merge_ml.py",
    "src/05_merge_context.py",
    "src/06_ml_baseline.py",
    "src/07_data_quality_scorecard.py",
    "src/08_segmentation_outliers.py",
]


def run_step(step: str) -> None:
    script = REPO_ROOT / step
    if not script.exists():
        raise SystemExit(f"Missing {script}")
    print(f"[v1] Running {step}...")
    env = dict(os.environ)
    env["MHP_VERSION"] = "v1"
    subprocess.run([PYTHON, str(script)], cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    for step in STEPS:
        run_step(step)
    print("[v1] Pipeline complete.")


if __name__ == "__main__":
    main()
