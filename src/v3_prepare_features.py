#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


FEATURE_COLS = [
    "suicide_rate",
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_v1() -> pd.DataFrame:
    source_path = REPO_ROOT / "v1" / "data_clean" / "merged_ml_country.csv"
    if not source_path.exists():
        raise SystemExit(f"Missing {source_path}. Build v1 outputs first.")

    df = pd.read_csv(source_path)
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")].copy()

    df = coerce_numeric(
        df,
        [
            "age_standardized_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "gbd_selfharm_death_rate_male",
            "gbd_selfharm_death_rate_female",
        ],
    )
    df["selfharm_both"] = df[
        ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
    ].mean(axis=1)

    group_cols = ["iso3", "location_name", "region_name", "income_group"]
    agg = (
        df[group_cols + [
            "age_standardized_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "selfharm_both",
        ]]
        .groupby(group_cols, as_index=False)
        .mean()
    )

    agg = agg.rename(
        columns={
            "age_standardized_suicide_rate_2021": "suicide_rate",
            "gbd_depression_dalys_rate_both": "depression_dalys_rate",
            "gbd_addiction_death_rate_both": "addiction_death_rate",
            "selfharm_both": "selfharm_death_rate",
        }
    )
    agg["sex_name"] = "Both"
    agg["year"] = 2021
    return agg


def prepare_v2() -> pd.DataFrame:
    source_path = REPO_ROOT / "v2" / "data_clean" / "synth_country_year.csv"
    if not source_path.exists():
        raise SystemExit(f"Missing {source_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(source_path)
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")].copy()
    df = df[df["year"] == 2021].copy()
    if "age_name" in df.columns:
        df = df[df["age_name"] == "All ages"].copy()

    df = coerce_numeric(df, FEATURE_COLS)
    keep_cols = [
        "iso3",
        "location_name",
        "region_name",
        "income_group",
        "sex_name",
        "year",
    ] + FEATURE_COLS
    df = df[keep_cols]
    return df


def write_summary(path: Path, label: str, df: pd.DataFrame) -> list[str]:
    lines = [
        f"## {label}",
        f"- Rows: {len(df):,}",
        f"- Countries: {df['iso3'].nunique():,}",
        f"- Sexes: {', '.join(sorted(df['sex_name'].unique()))}",
        f"- Years: {df['year'].min()}-{df['year'].max()}",
    ]
    missing = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    if missing.max() > 0:
        lines.append("- Missingness (% per feature):")
        for col, pct in missing.items():
            lines.append(f"  - {col}: {pct:.2%}")
    else:
        lines.append("- Missingness: 0% on core features")
    return lines


def main() -> None:
    ensure_dirs()
    if VERSION != "v3":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {DATA_CLEAN}")

    v1 = prepare_v1()
    v1 = v1.dropna(subset=FEATURE_COLS)
    v1_path = DATA_CLEAN / "v3_features_v1.csv"
    v1.to_csv(v1_path, index=False)

    v2 = prepare_v2()
    v2 = v2.dropna(subset=FEATURE_COLS)
    v2_path = DATA_CLEAN / "v3_features_v2.csv"
    v2.to_csv(v2_path, index=False)

    summary_lines = [
        "# v3 Feature Tables Summary",
        "",
    ]
    summary_lines.extend(write_summary(v1_path, "v1 source", v1))
    summary_lines.append("")
    summary_lines.extend(write_summary(v2_path, "v2 source", v2))
    summary_path = REPORT_DIR / "v3_feature_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote {len(v1)} rows to {v1_path}")
    print(f"[{VERSION}] Wrote {len(v2)} rows to {v2_path}")
    print(f"[{VERSION}] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
