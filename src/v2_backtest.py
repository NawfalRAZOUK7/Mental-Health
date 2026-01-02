#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


LAG_COUNT = 5
MIN_TRAIN = 5


def main() -> None:
    ensure_dirs()
    data_path = DATA_CLEAN / "synth_region_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df = df[df["sex_name"] == "Both"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["suicide_rate"] = pd.to_numeric(df["suicide_rate"], errors="coerce")
    df = df.dropna(subset=["year"])

    filled = []
    for region, group in df.groupby("region_name", sort=False):
        group = group.sort_values("year").copy()
        if group["suicide_rate"].isna().all():
            continue
        group["suicide_rate"] = group["suicide_rate"].interpolate(limit_direction="both")
        group["suicide_rate"] = group["suicide_rate"].fillna(group["suicide_rate"].median())
        filled.append(group)

    if not filled:
        raise SystemExit("No usable region series after filling missing values.")

    df = pd.concat(filled, ignore_index=True)

    pred_rows = []
    metrics_rows = []
    lag_cols = [f"lag_{i}" for i in range(1, LAG_COUNT + 1)]

    for region, group in df.groupby("region_name", sort=False):
        group = group.sort_values("year").copy()
        for i in range(1, LAG_COUNT + 1):
            group[f"lag_{i}"] = group["suicide_rate"].shift(i)

        group = group.dropna(subset=lag_cols + ["suicide_rate"])
        if len(group) <= MIN_TRAIN:
            continue

        preds = []
        actuals = []

        for idx in range(MIN_TRAIN, len(group)):
            train = group.iloc[:idx]
            test = group.iloc[idx: idx + 1]

            X_train = train[lag_cols].to_numpy()
            y_train = train["suicide_rate"].to_numpy()
            X_test = test[lag_cols].to_numpy()

            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])

            preds.append(pred)
            actuals.append(float(test["suicide_rate"].iloc[0]))
            pred_rows.append(
                {
                    "region_name": region,
                    "year": int(test["year"].iloc[0]),
                    "actual": float(test["suicide_rate"].iloc[0]),
                    "predicted": pred,
                }
            )

        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds) if len(actuals) > 1 else float("nan")
        metrics_rows.append(
            {
                "region_name": region,
                "mae": mae,
                "r2": r2,
                "rows": len(actuals),
            }
        )

    pred_cols = ["region_name", "year", "actual", "predicted"]
    pred_df = pd.DataFrame(pred_rows, columns=pred_cols)
    pred_path = REPORT_DIR / "v2_backtest_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_cols = ["region_name", "mae", "r2", "rows"]
    metrics_df = pd.DataFrame(metrics_rows, columns=metrics_cols)
    if not pred_df.empty:
        overall_mae = mean_absolute_error(pred_df["actual"], pred_df["predicted"])
        overall_r2 = r2_score(pred_df["actual"], pred_df["predicted"])
        overall = pd.DataFrame(
            [
                {
                    "region_name": "Overall",
                    "mae": overall_mae,
                    "r2": overall_r2,
                    "rows": len(pred_df),
                }
            ]
        )
        metrics_df = pd.concat([metrics_df, overall], ignore_index=True)

    metrics_path = REPORT_DIR / "v2_backtest_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[{VERSION}] Wrote backtest predictions to {pred_path}")
    print(f"[{VERSION}] Wrote backtest metrics to {metrics_path}")


if __name__ == "__main__":
    main()
