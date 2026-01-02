#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


YEARS = np.arange(2000, 2024)
SEXES = ["Male", "Female"]
GENERATOR_VERSION = "1.0"
SEED = 42

AGE_MULTIPLIER = {
    "<20 years": 0.6,
    "20-24 years": 0.9,
    "25+ years": 1.1,
}

SEX_MULTIPLIER = {
    "Male": 1.6,
    "Female": 0.6,
}

AGE_SHARE = {
    "<20 years": 0.3,
    "20-24 years": 0.1,
    "25+ years": 0.6,
}

SEX_SHARE = {
    "Male": 0.505,
    "Female": 0.495,
}

INCOME_BASE = {
    "HIC": 50_000_000,
    "UMI": 30_000_000,
    "LMI": 18_000_000,
    "LI": 8_000_000,
}

REGION_BIAS = {
    "African Region": 0.15,
    "Region of the Americas": 0.05,
    "Eastern Mediterranean Region": 0.08,
    "European Region": -0.05,
    "South-East Asia Region": 0.07,
    "Western Pacific Region": -0.02,
}


def weighted_mean(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    df = df[df["population"].notna() & (df["population"] > 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=group_cols + value_cols + ["population"])

    grouped = []
    for keys, chunk in df.groupby(group_cols, sort=False):
        weights = chunk["population"].to_numpy()
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["population"] = weights.sum()
        for col in value_cols:
            values = chunk[col].to_numpy()
            mask = ~np.isnan(values)
            if not mask.any():
                row[col] = np.nan
            else:
                row[col] = np.average(values[mask], weights=weights[mask])
        grouped.append(row)
    return pd.DataFrame(grouped)


def build_population_params(base: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for iso3, group in base.groupby("iso3"):
        income = str(group["income_group"].iloc[0]) if not group.empty else "Unknown"
        scale = INCOME_BASE.get(income, 25_000_000)
        pop_base = rng.lognormal(mean=np.log(scale), sigma=0.6)
        pop_base = float(np.clip(pop_base, 200_000, 1_500_000_000))
        growth = float(rng.normal(0.01, 0.004))
        params[str(iso3)] = {"base": pop_base, "growth": growth}
    return params


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {DATA_CLEAN}")

    source_path = REPO_ROOT / "v1" / "data_clean" / "merged_ml_country.csv"
    if not source_path.exists():
        raise SystemExit(f"Missing {source_path}. Build v1 outputs first.")

    df = pd.read_csv(source_path)
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")].copy()
    df["region_name"] = df["region_name"].fillna("Unknown")
    df["income_group"] = df["income_group"].fillna("Unknown")

    df["gbd_selfharm_death_rate_both"] = df[
        ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
    ].mean(axis=1)

    numeric_cols = [
        "age_standardized_suicide_rate_2021",
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_addiction_death_rate_male",
        "gbd_addiction_death_rate_female",
        "gbd_selfharm_death_rate_both",
        "gbd_selfharm_death_rate_male",
        "gbd_selfharm_death_rate_female",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    group_cols = ["iso3", "location_name", "region_name", "income_group", "age_name"]
    agg_cols = {col: "mean" for col in numeric_cols}
    base = (
        df[group_cols + list(agg_cols.keys())]
        .groupby(group_cols, as_index=False)
        .agg(agg_cols)
    )

    rng = np.random.default_rng(SEED)
    pop_params = build_population_params(base, rng)

    records = []
    for row in base.itertuples(index=False):
        region_bias = REGION_BIAS.get(row.region_name, 0.0)
        latent = float(rng.normal(region_bias, 0.6))
        trend = float(rng.normal(0.003 + region_bias * 0.001, 0.004))
        age_mult = AGE_MULTIPLIER.get(row.age_name, 1.0)

        base_suicide = float(row.age_standardized_suicide_rate_2021) * age_mult
        base_depression = float(row.gbd_depression_dalys_rate_both) * age_mult

        pop_base = pop_params.get(str(row.iso3), {"base": 25_000_000, "growth": 0.01})

        for year in YEARS:
            t = year - 2021
            trend_factor = max(0.4, 1 + trend * t)

            sex_records = []
            for sex in SEXES:
                sex_mult = SEX_MULTIPLIER.get(sex, 1.0)
                noise = float(rng.normal(0, 0.05))

                if sex == "Male":
                    base_addiction = float(row.gbd_addiction_death_rate_male)
                    base_selfharm = float(row.gbd_selfharm_death_rate_male)
                    dep_mult = 1.05
                else:
                    base_addiction = float(row.gbd_addiction_death_rate_female)
                    base_selfharm = float(row.gbd_selfharm_death_rate_female)
                    dep_mult = 0.95

                depression = (
                    base_depression * dep_mult * trend_factor * (1 + 0.08 * latent) * (1 + 0.05 * noise)
                )
                addiction = (
                    base_addiction * trend_factor * (1 + 0.06 * latent) * (1 + 0.05 * noise)
                )
                selfharm = (
                    base_selfharm * trend_factor * (1 + 0.07 * latent) * (1 + 0.05 * noise)
                )

                suicide = (
                    base_suicide
                    * sex_mult
                    * trend_factor
                    * (1 + 0.1 * latent)
                    + 0.12 * depression
                    + 0.1 * selfharm
                    + 0.06 * addiction
                    + rng.normal(0, 0.8)
                )

                population = pop_base["base"] * (1 + pop_base["growth"]) ** (year - 2000)
                population *= AGE_SHARE.get(row.age_name, 0.33)
                population *= SEX_SHARE.get(sex, 1.0)

                sex_records.append(
                    {
                        "iso3": row.iso3,
                        "location_name": row.location_name,
                        "region_name": row.region_name,
                        "income_group": row.income_group,
                        "sex_name": sex,
                        "age_name": row.age_name,
                        "year": year,
                        "suicide_rate": max(0.0, float(suicide)),
                        "depression_dalys_rate": max(0.0, float(depression)),
                        "addiction_death_rate": max(0.0, float(addiction)),
                        "selfharm_death_rate": max(0.0, float(selfharm)),
                        "population": max(0.0, float(population)),
                        "is_synthetic": 1,
                        "generator_version": GENERATOR_VERSION,
                        "seed": SEED,
                    }
                )

            male = sex_records[0]
            female = sex_records[1]
            both_pop = male["population"] + female["population"]
            if both_pop > 0:
                both = {
                    "iso3": male["iso3"],
                    "location_name": male["location_name"],
                    "region_name": male["region_name"],
                    "income_group": male["income_group"],
                    "sex_name": "Both",
                    "age_name": male["age_name"],
                    "year": year,
                    "suicide_rate": (male["suicide_rate"] * male["population"] + female["suicide_rate"] * female["population"]) / both_pop,
                    "depression_dalys_rate": (male["depression_dalys_rate"] * male["population"] + female["depression_dalys_rate"] * female["population"]) / both_pop,
                    "addiction_death_rate": (male["addiction_death_rate"] * male["population"] + female["addiction_death_rate"] * female["population"]) / both_pop,
                    "selfharm_death_rate": (male["selfharm_death_rate"] * male["population"] + female["selfharm_death_rate"] * female["population"]) / both_pop,
                    "population": both_pop,
                    "is_synthetic": 1,
                    "generator_version": GENERATOR_VERSION,
                    "seed": SEED,
                }
                sex_records.append(both)

            records.extend(sex_records)

    synth = pd.DataFrame(records)
    feature_cols = [
        "suicide_rate",
        "depression_dalys_rate",
        "addiction_death_rate",
        "selfharm_death_rate",
    ]

    for col in feature_cols:
        mask = rng.random(len(synth)) < 0.03
        synth.loc[mask, col] = np.nan

    z_features = synth[feature_cols].fillna(synth[feature_cols].median())
    z = (z_features - z_features.mean()) / z_features.std(ddof=0)
    synth["risk_index"] = z.mean(axis=1)

    output_path = DATA_CLEAN / "synth_long.csv"
    synth.to_csv(output_path, index=False)

    country_year = weighted_mean(
        synth,
        ["iso3", "location_name", "region_name", "income_group", "sex_name", "year"],
        feature_cols + ["risk_index"],
    )
    country_year["age_name"] = "All ages"
    country_year["is_synthetic"] = 1
    country_year["generator_version"] = GENERATOR_VERSION
    country_year["seed"] = SEED
    country_year_path = DATA_CLEAN / "synth_country_year.csv"
    country_year.to_csv(country_year_path, index=False)

    region_year = weighted_mean(
        synth,
        ["region_name", "sex_name", "year"],
        feature_cols + ["risk_index"],
    )
    region_year["location_name"] = region_year["region_name"]
    region_year["iso3"] = ""
    region_year["income_group"] = ""
    region_year["age_name"] = "All ages"
    region_year["is_synthetic"] = 1
    region_year["generator_version"] = GENERATOR_VERSION
    region_year["seed"] = SEED
    region_year_path = DATA_CLEAN / "synth_region_year.csv"
    region_year.to_csv(region_year_path, index=False)

    notes = [
        "# Synthetic Generation Notes",
        "",
        "This dataset is synthetic and intended for demonstration only.",
        "",
        f"- Seed: {SEED}",
        f"- Generator version: {GENERATOR_VERSION}",
        f"- Years: {YEARS.min()}-{YEARS.max()}",
        "- Base distributions from v1 merged_ml_country.csv (WHO + GBD).",
        "- Missing base values are filled with median before synthesis.",
        "- Trends include region bias, latent risk, and Gaussian noise.",
        "- Age/sex multipliers and income-scaled population growth add variation.",
        "- Both-sex rows are population-weighted averages of male/female.",
        "- Country/region outputs are population-weighted aggregates from the long table.",
        "- Missingness injected: ~3% per feature column.",
        "- risk_index is the mean z-score of the core feature columns.",
        "",
        "Outputs:",
        f"- {output_path.relative_to(REPO_ROOT)}",
        f"- {country_year_path.relative_to(REPO_ROOT)}",
        f"- {region_year_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "synth_generation_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    dict_lines = [
        "# Synthetic Data Dictionary",
        "",
        "| Column | Definition | Unit |",
        "| --- | --- | --- |",
        "| iso3 | ISO3 country code | n/a |",
        "| location_name | Country name | n/a |",
        "| region_name | WHO region | n/a |",
        "| income_group | Income group | n/a |",
        "| sex_name | Sex category | n/a |",
        "| age_name | Age group | n/a |",
        "| year | Year | year |",
        "| suicide_rate | Synthetic suicide rate | per 100k |",
        "| depression_dalys_rate | Synthetic depression DALYs rate | per 100k |",
        "| addiction_death_rate | Synthetic addiction deaths rate | per 100k |",
        "| selfharm_death_rate | Synthetic self-harm deaths rate | per 100k |",
        "| risk_index | Composite risk z-score | z-score |",
        "| population | Synthetic population | count |",
        "| is_synthetic | Synthetic flag | 0/1 |",
        "| generator_version | Generator version | n/a |",
        "| seed | Random seed | n/a |",
    ]
    dict_path = REPORT_DIR / "synth_data_dictionary.md"
    dict_path.write_text("\n".join(dict_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote {len(synth)} rows to {output_path}")
    print(f"[{VERSION}] Wrote {len(country_year)} rows to {country_year_path}")
    print(f"[{VERSION}] Wrote {len(region_year)} rows to {region_year_path}")
    print(f"[{VERSION}] Wrote notes to {notes_path}")
    print(f"[{VERSION}] Wrote dictionary to {dict_path}")


if __name__ == "__main__":
    main()
