#!/usr/bin/env python3
"""Fetch and cache socioeconomic covariates from the World Bank Open Data API.

These covariates give the v1 model *independent* predictors of the suicide rate
(beyond the near-tautological self-harm death rate), enabling an honest analysis
of which socioeconomic factors are associated with national suicide rates.

Run locally (needs internet):
    python src/10_enrich_covariates.py

Output:
    data_raw/worldbank_covariates.csv  (one row per country, latest available year)

The script is idempotent and caches results. If the API is unreachable it exits
with a clear message and leaves any existing cache in place.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request

import pandas as pd

from project_paths import DATA_RAW

# World Bank indicator code -> friendly column name.
INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "SH.XPD.CHEX.PC.CD": "health_exp_per_capita_usd",
    "SH.ALC.PCAP.LI": "alcohol_litres_per_capita",
    "SI.POV.GINI": "gini_index",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "SP.DYN.LE00.IN": "life_expectancy_years",
}

BASE = "https://api.worldbank.org/v2/country/all/indicator/{code}"
# Pull recent years and keep the latest non-null per country.
PARAMS = "?format=json&mrv=6&per_page=20000"
# The World Bank firewall rejects the default urllib User-Agent with HTTP 400,
# so send a normal browser-like UA.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mental-health-viz/1.0)"}


def fetch_indicator(code: str) -> pd.DataFrame:
    url = BASE.format(code=code) + PARAMS
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["iso3", "year", "value"])
    rows = [
        {
            "iso3": rec.get("countryiso3code"),
            "year": int(rec["date"]),
            "value": rec.get("value"),
        }
        for rec in payload[1]
        if rec.get("countryiso3code") and rec.get("value") is not None
    ]
    return pd.DataFrame(rows)


def latest_per_country(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["iso3", col])
    df = df.sort_values("year")
    latest = df.groupby("iso3", as_index=False).last()
    return latest.rename(columns={"value": col})[["iso3", col]]


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "worldbank_covariates.csv"

    merged: pd.DataFrame | None = None
    for code, name in INDICATORS.items():
        try:
            raw = fetch_indicator(code)
        except Exception as exc:  # noqa: BLE001 - network/parse errors -> clear message
            print(f"FAILED to fetch {code} ({name}): {exc}", file=sys.stderr)
            if out_path.exists():
                print(f"Keeping existing cache at {out_path}")
                return
            raise SystemExit(
                "Could not reach the World Bank API and no cache exists. "
                "Check your internet connection and retry."
            ) from exc
        col = latest_per_country(raw, name)
        merged = col if merged is None else merged.merge(col, on="iso3", how="outer")
        print(f"  {name:28s} {col[name].notna().sum():>3d} countries")
        time.sleep(0.3)  # be polite to the API

    assert merged is not None
    merged = merged.sort_values("iso3").reset_index(drop=True)

    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nWrote {len(merged)} countries x {len(INDICATORS)} covariates -> {out_path}")


if __name__ == "__main__":
    main()
