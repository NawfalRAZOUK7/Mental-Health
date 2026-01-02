#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs


Z_THRESHOLD = 2.0


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
        out_df = pd.DataFrame(
            [],
            columns=["region_name", "year", "suicide_rate", "diff", "zscore"],
        )
        out_path = REPORT_DIR / "v2_changepoints.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[{VERSION}] Wrote changepoints to {out_path}")
        return

    df = pd.concat(filled, ignore_index=True)

    rows = []
    for region, group in df.groupby("region_name", sort=False):
        group = group.sort_values("year").copy()
        group["diff"] = group["suicide_rate"].diff()
        diff = group["diff"].dropna()
        if diff.std(ddof=0) == 0 or diff.empty:
            continue
        zscores = (group["diff"] - diff.mean()) / diff.std(ddof=0)
        group["zscore"] = zscores
        flagged = group[group["zscore"].abs() >= Z_THRESHOLD]
        for _, row in flagged.iterrows():
            rows.append(
                {
                    "region_name": region,
                    "year": int(row["year"]),
                    "suicide_rate": float(row["suicide_rate"]),
                    "diff": float(row["diff"]),
                    "zscore": float(row["zscore"]),
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=["region_name", "year", "suicide_rate", "diff", "zscore"],
    )
    out_path = REPORT_DIR / "v2_changepoints.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[{VERSION}] Wrote changepoints to {out_path}")


if __name__ == "__main__":
    main()
