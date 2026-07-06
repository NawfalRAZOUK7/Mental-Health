#!/usr/bin/env python3
"""Run the advanced real-data ML + data-mining suite (v1).

Requires: pip install -r requirements-advanced.txt
Also recommended first: python src/10_enrich_covariates.py  (World Bank covariates)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STEPS = [
    "src/12_ml_advanced.py",       # LightGBM + Optuna nested CV + SHAP + conformal
    "src/13_hierarchical_model.py",  # mixed-effects + ICC
    "src/14_umap_embedding.py",    # UMAP country embedding
    "src/15_country_network.py",   # similarity network + communities
    "src/16_subgroup_rules.py",    # subgroup discovery + association rules
]


def main() -> None:
    env = dict(os.environ)
    env["MHP_VERSION"] = "v1"
    for step in STEPS:
        script = REPO_ROOT / step
        if not script.exists():
            raise SystemExit(f"Missing {script}")
        print(f"[advanced] Running {step} ...")
        subprocess.run([PYTHON, str(script)], cwd=REPO_ROOT, env=env, check=True)
    print("[advanced] Done. Optional deep global forecaster (synthetic panel):")
    print("           MHP_VERSION=v2 python src/v2_global_forecast.py")


if __name__ == "__main__":
    main()
