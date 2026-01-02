#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs


FEATURE_COLS = [
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]
TARGET_COL = "suicide_rate"
QUANTILES = [0.1, 0.5, 0.9]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["sex_name"] == "Both") & (df["year"] >= 2010)]
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    if df.empty:
        raise SystemExit("No data available for quantile regression.")

    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy()

    preds = {}
    for quantile in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            random_state=42,
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
        )
        model.fit(X, y)
        preds[quantile] = model.predict(X)

    pred_df = df[
        [
            "iso3",
            "location_name",
            "region_name",
            "income_group",
            "sex_name",
            "year",
            TARGET_COL,
        ]
    ].copy()
    pred_df["q10"] = preds[0.1]
    pred_df["q50"] = preds[0.5]
    pred_df["q90"] = preds[0.9]

    pred_path = REPORT_DIR / "v2_quantile_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_rows = []
    for quantile in QUANTILES:
        q_pred = preds[quantile]
        metrics_rows.append(
            {
                "quantile": quantile,
                "pinball_loss": pinball_loss(y, q_pred, quantile),
                "mae": mean_absolute_error(y, q_pred) if quantile == 0.5 else np.nan,
            }
        )
    coverage = float(np.mean((y >= preds[0.1]) & (y <= preds[0.9])))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["interval_coverage_10_90"] = coverage
    metrics_path = REPORT_DIR / "v2_quantile_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[{VERSION}] Wrote quantile predictions to {pred_path}")


if __name__ == "__main__":
    main()
