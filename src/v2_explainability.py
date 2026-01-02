#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.inspection import partial_dependence, permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
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


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df = df[(df["sex_name"] == "Both") & (df["year"] >= 2010)]
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    if df.empty:
        raise SystemExit("No data available for explainability.")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].to_numpy()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)

    perm = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=20,
        random_state=42,
        scoring="r2",
    )
    perm_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    perm_path = REPORT_DIR / "v2_perm_importance.csv"
    perm_df.to_csv(perm_path, index=False)

    top_features = perm_df["feature"].head(3).tolist()
    pdp_rows = []
    for feature in top_features:
        feat_idx = FEATURE_COLS.index(feature)
        pd_result = partial_dependence(pipeline, X, [feat_idx], grid_resolution=20)
        grid_key = "grid_values" if "grid_values" in pd_result else "values"
        values = np.array(pd_result[grid_key][0], dtype=float)
        avg = np.squeeze(np.array(pd_result["average"], dtype=float))
        for value, pdp_val in zip(values, avg):
            pdp_rows.append(
                {
                    "feature": feature,
                    "feature_value": float(value),
                    "pdp": float(pdp_val),
                }
            )

    pdp_df = pd.DataFrame(pdp_rows)
    pdp_path = REPORT_DIR / "v2_partial_dependence.csv"
    pdp_df.to_csv(pdp_path, index=False)

    print(f"[{VERSION}] Wrote permutation importance to {perm_path}")
    print(f"[{VERSION}] Wrote partial dependence to {pdp_path}")


if __name__ == "__main__":
    main()
