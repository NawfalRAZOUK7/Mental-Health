#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as exc:
    raise SystemExit(
        "mlxtend is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, VERSION, ensure_dirs


FEATURE_COLS = [
    "suicide_rate",
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]
BUCKETS = ["low", "mid", "high"]


def bin_series(series: pd.Series) -> pd.Series:
    return pd.qcut(series, q=3, labels=BUCKETS, duplicates="drop")


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(data_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] == 2023) & (df["sex_name"] == "Both")].copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    if df.empty:
        raise SystemExit("No data available for association rules.")

    binned = {}
    for col in FEATURE_COLS:
        binned[col] = bin_series(df[col])

    encoded_rows = []
    for idx in df.index:
        row = {}
        for col in FEATURE_COLS:
            bucket = str(binned[col].loc[idx])
            if bucket == "nan":
                continue
            row[f"{col}_{bucket}"] = 1
        encoded_rows.append(row)

    one_hot = pd.DataFrame(encoded_rows).fillna(0).astype(bool)
    if one_hot.empty:
        raise SystemExit("No transactions after binning.")

    support_levels = [0.08, 0.06, 0.05, 0.03]
    confidence_levels = [0.25, 0.2, 0.15]
    rules = pd.DataFrame()
    chosen_support = None
    chosen_conf = None

    for support in support_levels:
        frequent = apriori(one_hot, min_support=support, use_colnames=True)
        if frequent.empty:
            continue
        for conf in confidence_levels:
            candidate = association_rules(
                frequent,
                metric="confidence",
                min_threshold=conf,
            )
            if not candidate.empty:
                rules = candidate
                chosen_support = support
                chosen_conf = conf
                if len(rules) >= 10:
                    break
        if not rules.empty:
            break

    if rules.empty:
        raise SystemExit("No association rules found. Adjust min_support.")

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules["support_threshold"] = chosen_support
    rules["confidence_threshold"] = chosen_conf

    output_cols = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "support_threshold",
        "confidence_threshold",
    ]
    rules[output_cols].to_csv(REPORT_DIR / "v2_assoc_rules.csv", index=False)

    print(f"[{VERSION}] Wrote association rules to {REPORT_DIR / 'v2_assoc_rules.csv'}")


if __name__ == "__main__":
    main()
