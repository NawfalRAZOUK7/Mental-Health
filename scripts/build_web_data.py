#!/usr/bin/env python3
"""Generate data + assets for the Next.js web app (web/).

Writes web/data/predictions.json (baked-in model predictions for every country)
and copies the key figures into web/public/assets/. Run before building the site:

    python scripts/build_web_data.py
    cd web && npm install && npm run build   # static export -> web/out/
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MHP_VERSION", "v1")

FIGURES = [
    "fig_v1_shap_summary.png",
    "fig_v1_umap_countries.png",
    "fig_v1_country_network.png",
    "fig_v1_source_agreement.png",
]


def main() -> None:
    from api.model import Predictor

    predictor = Predictor()
    preds = []
    for c in predictor.countries:
        r = predictor.predict_country(c["iso3"])
        if r is None:
            continue
        preds.append({
            "iso3": r["iso3"], "name": r["name"],
            "pred": r["predicted_suicide_rate"],
            "lower": max(0.0, r["lower_90"]),
            "upper": r["upper_90"],
            "actual": r["actual_suicide_rate"],
        })

    web = REPO_ROOT / "web"
    (web / "data").mkdir(parents=True, exist_ok=True)
    (web / "data" / "predictions.json").write_text(
        json.dumps(preds, separators=(",", ":")), encoding="utf-8"
    )

    assets = web / "public" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig_src = REPO_ROOT / "report_latex" / "figures"
    copied = sum(
        bool(shutil.copyfile(fig_src / n, assets / n))
        for n in FIGURES if (fig_src / n).exists()
    )

    print(f"[web] Wrote {len(preds)} predictions to web/data/predictions.json")
    print(f"[web] Copied {copied}/{len(FIGURES)} figures to web/public/assets/")


if __name__ == "__main__":
    main()
