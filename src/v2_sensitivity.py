#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs


def main() -> None:
    ensure_dirs()
    coeff_path = REPORT_DIR / "v2_model_coeffs.csv"
    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not coeff_path.exists() or not data_path.exists():
        raise SystemExit("Missing model coefficients or synthetic data. Run v2_analytics first.")

    coeffs = pd.read_csv(coeff_path)
    coeffs["coef"] = pd.to_numeric(coeffs["coef"], errors="coerce")
    intercept = float(coeffs[coeffs["feature"] == "_intercept"]["coef"].iloc[0])
    coeffs = coeffs[coeffs["feature"] != "_intercept"].copy()
    coeffs["mean"] = pd.to_numeric(coeffs["mean"], errors="coerce")
    coeffs["scale"] = pd.to_numeric(coeffs["scale"], errors="coerce")

    df = pd.read_csv(data_path)
    df = df[(df["sex_name"] == "Both") & (df["year"] == df["year"].max())]
    feature_cols = coeffs["feature"].tolist()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    baseline = df[feature_cols].median()
    base_pred = intercept
    for _, row in coeffs.iterrows():
        feature = row["feature"]
        mean = float(row["mean"])
        scale = float(row["scale"]) if float(row["scale"]) != 0 else 1.0
        val = float(baseline[feature])
        base_pred += float(row["coef"]) * ((val - mean) / scale)

    rows = []
    for _, row in coeffs.iterrows():
        feature = row["feature"]
        mean = float(row["mean"])
        scale = float(row["scale"]) if float(row["scale"]) != 0 else 1.0
        coef = float(row["coef"])
        base_val = float(baseline[feature])
        new_val = base_val * 1.10
        delta = coef * ((new_val - base_val) / scale)
        new_pred = base_pred + delta
        pct_change = (delta / base_pred * 100.0) if base_pred else float("nan")
        rows.append(
            {
                "feature": feature,
                "base_value": base_val,
                "new_value": new_val,
                "delta_prediction": delta,
                "pct_change_prediction": pct_change,
                "base_prediction": base_pred,
                "new_prediction": new_pred,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("pct_change_prediction", ascending=False)
    out_path = REPORT_DIR / "v2_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[{VERSION}] Wrote sensitivity outputs to {out_path}")


if __name__ == "__main__":
    main()
