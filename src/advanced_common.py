#!/usr/bin/env python3
"""Shared loader for the advanced v1 (real-data) ML / data-mining scripts.

Produces one row per country with:
  - target:  suicide_rate (WHO age-standardized)
  - epi features: depression_dalys_rate, addiction_death_rate
  - socioeconomic covariates (World Bank) when data_raw/worldbank_covariates.csv exists
  - categoricals: region_name, income_group

The IHME self-harm rate is deliberately excluded as a predictor (it is ~ suicide;
see 09_source_agreement.py). It is returned only as a reference column.
"""
from __future__ import annotations

import pandas as pd

from project_paths import DATA_CLEAN, DATA_RAW

EPI_FEATURES = ["depression_dalys_rate", "addiction_death_rate"]
CAT_FEATURES = ["region_name", "income_group"]
TARGET = "suicide_rate"


def load_enriched() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (dataframe, numeric_feature_cols, categorical_feature_cols)."""
    path = DATA_CLEAN / "merged_ml_country.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 04_merge_ml.py first.")
    df = pd.read_csv(path)
    for col in [
        "age_standardized_suicide_rate_2021",
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_selfharm_death_rate_male",
        "gbd_selfharm_death_rate_female",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["selfharm"] = df[["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]].mean(axis=1)

    g = (
        df.groupby(["iso3", "location_name", "region_name", "income_group"], as_index=False)
        .agg(
            suicide_rate=("age_standardized_suicide_rate_2021", "mean"),
            depression_dalys_rate=("gbd_depression_dalys_rate_both", "mean"),
            addiction_death_rate=("gbd_addiction_death_rate_both", "mean"),
            selfharm_rate=("selfharm", "mean"),
        )
        .dropna(subset=[TARGET, *EPI_FEATURES])
    )

    num_features = list(EPI_FEATURES)
    cov_path = DATA_RAW / "worldbank_covariates.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        cov_cols = [c for c in cov.columns if c != "iso3"]
        g = g.merge(cov, on="iso3", how="left")
        num_features += cov_cols

    g[CAT_FEATURES] = g[CAT_FEATURES].fillna("Unknown")
    return g.reset_index(drop=True), num_features, list(CAT_FEATURES)
