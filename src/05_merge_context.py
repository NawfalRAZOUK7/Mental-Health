#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from project_paths import DATA_CLEAN, VERSION

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


METRICS_WITH_SCALE = {"Number", "Percent", "Rate"}
RATE_PER_100K = 100000.0


def load_gbd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(str)
    return df


def load_preferred_countries(data_clean: Path) -> pd.DataFrame:
    mapping_path = data_clean / "country_iso3_mapping.csv"
    if not mapping_path.exists():
        raise SystemExit(f"Missing {mapping_path}. Run 01_country_mapping.py first.")

    mapping = pd.read_csv(mapping_path)
    allowed: set[tuple[str, str]] = set()
    preferred: dict[str, str] = {}

    for _, row in mapping[mapping["source_type"] == "WHO"][["iso3", "name"]].dropna().iterrows():
        allowed.add((row["iso3"], row["name"]))
        preferred.setdefault(row["iso3"], row["name"])

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

    for iso3, name in allowed:
        preferred.setdefault(iso3, name)

    preferred_df = pd.DataFrame(
        [{"iso3": iso3, "location_name": name} for iso3, name in preferred.items()]
    )
    return preferred_df


def filter_to_preferred_locations(df: pd.DataFrame, preferred_df: pd.DataFrame) -> pd.DataFrame:
    if preferred_df.empty:
        return df
    return df.merge(preferred_df, on=["iso3", "location_name"], how="inner")


def to_category(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_who_region_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, usecols=["iso3", "region_name"])
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]
    return dict(zip(df["iso3"], df["region_name"]))


def filter_country_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["iso3"].notna() & (df["iso3"].astype(str) != "")].copy()


def attach_region_fields(df: pd.DataFrame, region_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["region_name"] = df["iso3"].map(region_map)
    df["location_type"] = "country"
    return df


def attach_weights(df: pd.DataFrame, weight_keys: list[str]) -> pd.DataFrame:
    number = (
        df[df["metric_name"] == "Number"][weight_keys + ["val", "upper", "lower"]]
        .groupby(weight_keys, as_index=False, observed=True)[["val", "upper", "lower"]]
        .mean()
        .rename(columns={"val": "number_val", "upper": "number_upper", "lower": "number_lower"})
    )
    rate = (
        df[df["metric_name"] == "Rate"][weight_keys + ["val"]]
        .groupby(weight_keys, as_index=False, observed=True)[["val"]]
        .mean()
        .rename(columns={"val": "rate_val"})
    )

    weights = number.merge(rate, on=weight_keys, how="inner")
    weights["population"] = weights["number_val"] / weights["rate_val"] * RATE_PER_100K
    weights = weights.replace([pd.NA, pd.NaT], None)
    weights = weights.replace([float("inf"), float("-inf")], pd.NA)
    weights = weights[weight_keys + ["number_val", "population"]]

    df = df.merge(weights, on=weight_keys, how="left")
    return df


def weighted_mean(
    df: pd.DataFrame, group_cols: list[str], weight_col: str, value_cols: list[str]
) -> pd.DataFrame:
    df = df[df[weight_col].notna() & (df[weight_col] > 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=group_cols + value_cols)

    denom = df.groupby(group_cols, observed=True)[weight_col].sum()
    weighted_df = df[group_cols + value_cols].copy()
    for col in value_cols:
        weighted_df[col] = weighted_df[col] * df[weight_col]
    numerator = weighted_df.groupby(group_cols, observed=True)[value_cols].sum()
    result = numerator.div(denom, axis=0).reset_index()
    return result


def weighted_aggregate(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    results = []
    for metric in sorted(df["metric_name"].dropna().unique()):
        subset = df[df["metric_name"] == metric].copy()
        if subset.empty:
            continue
        if metric == "Number":
            agg = subset.groupby(group_cols, as_index=False, observed=True)[value_cols].sum()
        elif metric == "Rate":
            agg = weighted_mean(subset, group_cols, "population", value_cols)
        elif metric == "Percent":
            agg = weighted_mean(subset, group_cols, "number_val", value_cols)
        else:
            agg = weighted_mean(subset, group_cols, "population", value_cols)
        agg["metric_name"] = metric
        results.append(agg)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def add_both_sexes_weighted(
    df: pd.DataFrame, weight_keys: list[str], group_cols: list[str]
) -> pd.DataFrame:
    base = df[df["sex_name"].isin(["Male", "Female"])].copy()
    if base.empty:
        return df

    base = attach_weights(base, weight_keys)
    both = weighted_aggregate(base, group_cols, ["val", "upper", "lower"])
    both["sex_name"] = "Both"
    return pd.concat([df, both], ignore_index=True)


def add_region_and_global_aggregates(
    df: pd.DataFrame,
    weight_keys: list[str],
    group_cols_region: list[str],
    group_cols_global: list[str],
    weights_attached: bool = False,
) -> pd.DataFrame:
    if not weights_attached:
        df = attach_weights(df, weight_keys)
    region_base = df.dropna(subset=["region_name"])
    region = weighted_aggregate(region_base, group_cols_region, ["val", "upper", "lower"])
    region["location_name"] = region["region_name"]
    region["iso3"] = ""
    region["location_type"] = "who_region"

    global_df = weighted_aggregate(df, group_cols_global, ["val", "upper", "lower"])
    global_df["region_name"] = "Global"
    global_df["location_name"] = "Global"
    global_df["iso3"] = ""
    global_df["location_type"] = "global"

    return pd.concat([df, region, global_df], ignore_index=True)


def build_population_map(allcauses: pd.DataFrame) -> pd.DataFrame:
    weight_keys = ["iso3", "sex_name", "age_name", "year", "measure_name"]
    allcauses = attach_weights(allcauses, weight_keys)
    pop = allcauses[["iso3", "sex_name", "age_name", "year", "population"]].dropna()
    return pop.drop_duplicates(subset=["iso3", "sex_name", "age_name", "year"])


def main() -> None:
    output_dir = DATA_CLEAN / "context_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    who_path = DATA_CLEAN / "who_2021_clean.csv"
    if not who_path.exists():
        raise SystemExit(f"Missing {who_path}. Run 02_clean_who.py first.")
    region_map = load_who_region_map(who_path)

    allcauses = load_gbd(DATA_CLEAN / "gbd_allcauses_clean.csv")
    allcauses = allcauses[
        (allcauses["cause_name"] == "All causes")
        & (allcauses["measure_name"] == "DALYs (Disability-Adjusted Life Years)")
        & (allcauses["metric_name"].isin(METRICS_WITH_SCALE))
    ]
    allcauses = allcauses[
        [
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    preferred_locations = load_preferred_countries(DATA_CLEAN)

    allcauses = filter_country_rows(allcauses)
    allcauses = filter_to_preferred_locations(allcauses, preferred_locations)
    allcauses = attach_region_fields(allcauses, region_map)
    allcauses = to_category(
        allcauses,
        ["iso3", "sex_name", "age_name", "year", "measure_name", "metric_name", "region_name"],
    )
    population_map = build_population_map(allcauses)

    allcauses = add_region_and_global_aggregates(
        allcauses,
        ["iso3", "sex_name", "age_name", "year", "measure_name"],
        ["region_name", "sex_name", "age_name", "year", "measure_name"],
        ["sex_name", "age_name", "year", "measure_name"],
    )
    allcauses = allcauses[
        [
            "location_type",
            "region_name",
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    allcauses_path = output_dir / "context_allcauses_trend.csv"
    allcauses.to_csv(allcauses_path, index=False)
    print(f"[{VERSION}] Wrote {len(allcauses)} rows to {allcauses_path}")

    big_categories = load_gbd(DATA_CLEAN / "gbd_big_categories_clean.csv")
    big_categories = big_categories[
        (big_categories["measure_name"] == "DALYs (Disability-Adjusted Life Years)")
        & (big_categories["metric_name"].isin(METRICS_WITH_SCALE))
        & (big_categories["year"] == "2023")
    ]
    big_categories = big_categories[
        [
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    big_categories = to_category(
        big_categories,
        [
            "iso3",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
        ],
    )
    big_categories = add_both_sexes_weighted(
        big_categories,
        ["location_name", "sex_name", "age_name", "year", "cause_name", "measure_name"],
        ["location_name", "age_name", "year", "cause_name", "measure_name"],
    )
    big_categories["region_name"] = big_categories["location_name"]
    big_categories["location_type"] = "gbd_aggregate"
    big_categories = to_category(
        big_categories,
        [
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "region_name",
        ],
    )
    big_categories = big_categories[
        [
            "location_type",
            "region_name",
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    big_categories_path = output_dir / "context_big_categories_2023.csv"
    big_categories.to_csv(big_categories_path, index=False)
    print(f"[{VERSION}] Wrote {len(big_categories)} rows to {big_categories_path}")

    probdeath = load_gbd(DATA_CLEAN / "gbd_prob_death_clean.csv")
    probdeath = probdeath[
        (probdeath["year"] == "2023")
        & (probdeath["age_name"] == "All ages")
        & (probdeath["cause_name"] == "All causes")
    ]
    probdeath = probdeath[
        [
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    probdeath = filter_country_rows(probdeath)
    probdeath = filter_to_preferred_locations(probdeath, preferred_locations)
    probdeath = probdeath.merge(
        population_map,
        on=["iso3", "sex_name", "age_name", "year"],
        how="left",
    )
    probdeath = attach_region_fields(probdeath, region_map)
    probdeath = to_category(
        probdeath,
        [
            "iso3",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "region_name",
        ],
    )
    probdeath = add_region_and_global_aggregates(
        probdeath,
        ["iso3", "sex_name", "age_name", "year", "measure_name", "cause_name"],
        ["region_name", "sex_name", "age_name", "year", "measure_name", "cause_name"],
        ["sex_name", "age_name", "year", "measure_name", "cause_name"],
        weights_attached=True,
    )
    probdeath = probdeath[
        [
            "location_type",
            "region_name",
            "iso3",
            "location_name",
            "sex_name",
            "age_name",
            "year",
            "cause_name",
            "measure_name",
            "metric_name",
            "val",
            "upper",
            "lower",
        ]
    ]
    probdeath_path = output_dir / "context_probdeath_2023.csv"
    probdeath.to_csv(probdeath_path, index=False)
    print(f"[{VERSION}] Wrote {len(probdeath)} rows to {probdeath_path}")


if __name__ == "__main__":
    main()
