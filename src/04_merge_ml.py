#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from project_paths import DATA_CLEAN, VERSION, ensure_dirs

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

try:
    import pycountry
except ImportError as exc:
    raise SystemExit(
        "pycountry is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


def load_who(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["sex_name"] == "Both sexes"].copy()
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]
    return df


def load_gbd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(str)
    return df


def load_allowed_locations(data_clean: Path) -> pd.DataFrame:
    mapping_path = data_clean / "country_iso3_mapping.csv"
    if not mapping_path.exists():
        raise SystemExit(f"Missing {mapping_path}. Run 01_country_mapping.py first.")

    mapping = pd.read_csv(mapping_path)
    allowed: set[tuple[str, str]] = set()
    for _, row in mapping[mapping["source_type"] == "WHO"][["iso3", "name"]].dropna().iterrows():
        allowed.add((row["iso3"], row["name"]))

    for _, row in mapping[mapping["match_type"] == "override"][["iso3", "name"]].dropna().iterrows():
        allowed.add((row["iso3"], row["name"]))

    gbd = mapping[mapping["source_type"] == "GBD"][["iso3", "name"]].dropna()
    for _, row in gbd.iterrows():
        try:
            country = pycountry.countries.lookup(row["name"])
        except LookupError:
            continue
        if country.alpha_3 == row["iso3"]:
            allowed.add((row["iso3"], row["name"]))

    allowed_df = pd.DataFrame(
        [{"iso3": iso3, "location_name": name} for iso3, name in allowed]
    )
    return allowed_df


def filter_to_allowed_locations(df: pd.DataFrame, allowed_df: pd.DataFrame) -> pd.DataFrame:
    if allowed_df.empty:
        return df
    return df.merge(allowed_df, on=["iso3", "location_name"], how="inner")


def aggregate_feature(
    df: pd.DataFrame, value_name: str, group_cols: list[str]
) -> pd.DataFrame:
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]
    grouped = df.groupby(group_cols, as_index=False)["val"].mean()
    return grouped.rename(columns={"val": value_name})


def pivot_sex_feature(
    df: pd.DataFrame,
    value_prefix: str,
    index_cols: list[str],
    sexes: list[str] | None = None,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=index_cols, columns="sex_name", values="val", aggfunc="mean"
    ).reset_index()
    pivot.columns = [
        f"{value_prefix}_{col.lower()}" if col not in index_cols else col
        for col in pivot.columns
    ]
    if sexes is None:
        sexes = ["Male", "Female", "Both"]
    keep_cols = set(index_cols)
    for sex in sexes:
        col = f"{value_prefix}_{sex.lower()}"
        keep_cols.add(col)
        if col not in pivot.columns:
            pivot[col] = pd.NA
    pivot = pivot[[col for col in pivot.columns if col in keep_cols]]
    return pivot


def main() -> None:
    ensure_dirs()
    who_path = DATA_CLEAN / "who_2021_clean.csv"
    if not who_path.exists():
        raise SystemExit(f"Missing {who_path}. Run 02_clean_who.py first.")

    allowed_locations = load_allowed_locations(DATA_CLEAN)

    addiction = load_gbd(DATA_CLEAN / "gbd_addiction_clean.csv")
    addiction = filter_to_allowed_locations(addiction, allowed_locations)
    addiction = addiction[
        (addiction["cause_name"] == "Substance use disorders")
        & (addiction["measure_name"] == "Deaths")
        & (addiction["metric_name"] == "Rate")
        & (addiction["year"] == "2023")
    ]
    addiction_feature = pivot_sex_feature(
        addiction, "gbd_addiction_death_rate", ["iso3"], sexes=["Male", "Female", "Both"]
    )

    depression = load_gbd(DATA_CLEAN / "gbd_depression_dalys_clean.csv")
    depression = filter_to_allowed_locations(depression, allowed_locations)
    depression = depression[
        (depression["cause_name"] == "Depressive disorders")
        & (depression["measure_name"] == "DALYs (Disability-Adjusted Life Years)")
        & (depression["metric_name"] == "Rate")
        & (depression["sex_name"] == "Both")
        & (depression["year"] == "2023")
    ]
    depression_feature = aggregate_feature(
        depression, "gbd_depression_dalys_rate_both", ["iso3", "age_name"]
    )

    selfharm = load_gbd(DATA_CLEAN / "gbd_selfharm_clean.csv")
    selfharm = filter_to_allowed_locations(selfharm, allowed_locations)
    selfharm = selfharm[
        (selfharm["cause_name"] == "Self-harm")
        & (selfharm["measure_name"] == "Deaths")
        & (selfharm["metric_name"] == "Rate")
        & (selfharm["sex_name"].isin(["Male", "Female"]))
        & (selfharm["year"] == "2023")
    ]
    selfharm_feature = pivot_sex_feature(
        selfharm, "gbd_selfharm_death_rate", ["iso3", "age_name"], sexes=["Male", "Female"]
    )

    age_groups = sorted(
        set(depression_feature["age_name"]).union(selfharm_feature["age_name"])
    )

    who_df = load_who(who_path)
    who_base = who_df[
        [
            "iso3",
            "location_name",
            "region_name",
            "year",
            "number_suicides_2021",
            "crude_suicide_rate_2021",
            "age_standardized_suicide_rate_2021",
            "data_quality",
            "income_group",
        ]
    ].copy()

    who_base = who_base.assign(_key=1).merge(
        pd.DataFrame({"age_name": age_groups, "_key": 1}), on="_key"
    )
    who_base = who_base.drop(columns="_key")

    merged = who_base.merge(addiction_feature, on="iso3", how="left")
    merged = merged.merge(depression_feature, on=["iso3", "age_name"], how="left")
    merged = merged.merge(selfharm_feature, on=["iso3", "age_name"], how="left")
    merged["gbd_year"] = 2023

    output_path = DATA_CLEAN / "merged_ml_country.csv"
    merged.to_csv(output_path, index=False)
    print(f"[{VERSION}] Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
