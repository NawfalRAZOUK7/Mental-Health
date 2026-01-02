#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs


FEATURE_COLS = [
    "suicide_rate",
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] == 2021) & (df["sex_name"] == "Both")].copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    if df.empty:
        raise SystemExit("No data available for KPI benchmarks.")

    rows = []
    for feature in FEATURE_COLS:
        series = df[feature].dropna()
        rows.append(
            {
                "feature": feature,
                "p10": float(series.quantile(0.10)),
                "median": float(series.quantile(0.50)),
                "p90": float(series.quantile(0.90)),
            }
        )

    out_path = REPORT_DIR / "v2_kpi_benchmarks.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[{VERSION}] Wrote KPI benchmarks to {out_path}")


if __name__ == "__main__":
    main()
