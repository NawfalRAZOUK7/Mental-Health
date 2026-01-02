#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs

CONTEXT_DIR = DATA_CLEAN / "context_tables"


DATASETS: dict[str, Path] = {
    "who_2021_clean": DATA_CLEAN / "who_2021_clean.csv",
    "gbd_depression_dalys_clean": DATA_CLEAN / "gbd_depression_dalys_clean.csv",
    "gbd_addiction_clean": DATA_CLEAN / "gbd_addiction_clean.csv",
    "gbd_selfharm_clean": DATA_CLEAN / "gbd_selfharm_clean.csv",
    "gbd_prob_death_clean": DATA_CLEAN / "gbd_prob_death_clean.csv",
    "gbd_allcauses_clean": DATA_CLEAN / "gbd_allcauses_clean.csv",
    "gbd_big_categories_clean": DATA_CLEAN / "gbd_big_categories_clean.csv",
    "merged_ml_country": DATA_CLEAN / "merged_ml_country.csv",
    "ml_baseline_features": DATA_CLEAN / "ml_baseline_features.csv",
    "context_allcauses_trend": CONTEXT_DIR / "context_allcauses_trend.csv",
    "context_big_categories_2023": CONTEXT_DIR / "context_big_categories_2023.csv",
    "context_probdeath_2023": CONTEXT_DIR / "context_probdeath_2023.csv",
}


def safe_year(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.1f}"


def count_missing(series: pd.Series) -> int:
    missing_count = int(series.isna().sum())
    if series.dtype == object:
        empty_count = (
            series[series.notna()].astype(str).str.strip().eq("").sum()
        )
        missing_count += int(empty_count)
    return missing_count


def summarize_dataset(name: str, df: pd.DataFrame) -> dict[str, object]:
    rows, cols = df.shape
    year_min = None
    year_max = None
    if "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce")
        year_min = years.min()
        year_max = years.max()

    iso3_missing_count = None
    iso3_missing_pct = None
    if "iso3" in df.columns:
        iso3_missing_count = count_missing(df["iso3"])
        iso3_missing_pct = round(
            (iso3_missing_count / rows) * 100, 2
        ) if rows else 0.0

    duplicate_rows = int(df.duplicated().sum())
    return {
        "dataset": name,
        "rows": rows,
        "columns": cols,
        "year_min": year_min,
        "year_max": year_max,
        "iso3_missing_count": iso3_missing_count,
        "iso3_missing_pct": iso3_missing_pct,
        "duplicate_rows": duplicate_rows,
    }


def markdown_table(df: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [header, divider]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return rows


def main() -> None:
    ensure_dirs()

    overview_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []

    for name, path in DATASETS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        overview_rows.append(summarize_dataset(name, df))

        rows = len(df)
        if rows == 0:
            continue
        for col in df.columns:
            missing_count = count_missing(df[col])
            if missing_count == 0:
                continue
            missingness_rows.append(
                {
                    "dataset": name,
                    "column": col,
                    "missing_count": missing_count,
                    "missing_pct": round((missing_count / rows) * 100, 2),
                }
            )

    overview_df = pd.DataFrame(overview_rows).sort_values("dataset")
    overview_path = REPORT_DIR / "data_quality_scorecard.csv"
    overview_df.to_csv(overview_path, index=False)

    missingness_df = pd.DataFrame(missingness_rows)
    if not missingness_df.empty:
        missingness_df = missingness_df.sort_values(
            ["missing_pct", "missing_count"], ascending=False
        )
    missingness_path = REPORT_DIR / "data_quality_missingness.csv"
    missingness_df.to_csv(missingness_path, index=False)

    who_quality_path = REPORT_DIR / "data_quality_who_data_quality.csv"
    who_path = DATASETS.get("who_2021_clean")
    if who_path and who_path.exists():
        who_df = pd.read_csv(who_path)
        counts = who_df["data_quality"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["data_quality", "count"]
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(2)
        counts.to_csv(who_quality_path, index=False)
    else:
        who_quality_path.write_text("data_quality,count,pct\n", encoding="utf-8")

    iso3_unmatched_path = REPORT_DIR / "data_quality_iso3_unmatched.csv"
    unmatched_path = DATA_CLEAN / "country_iso3_unmatched.csv"
    if unmatched_path.exists():
        unmatched_df = pd.read_csv(unmatched_path)
        if "source_type" in unmatched_df.columns:
            iso_counts = (
                unmatched_df["source_type"]
                .value_counts()
                .reset_index()
            )
            iso_counts.columns = ["source_type", "count"]
        else:
            iso_counts = pd.DataFrame({"source_type": ["unknown"], "count": [len(unmatched_df)]})
        iso_counts.to_csv(iso3_unmatched_path, index=False)
    else:
        iso3_unmatched_path.write_text("source_type,count\n", encoding="utf-8")

    overview_display = overview_df.copy()
    if not overview_display.empty:
        overview_display["year_min"] = overview_display["year_min"].apply(safe_year)
        overview_display["year_max"] = overview_display["year_max"].apply(safe_year)
        overview_display["iso3_missing_pct"] = overview_display["iso3_missing_pct"].apply(
            lambda x: "n/a" if pd.isna(x) else f"{x:.2f}%"
        )
        overview_display["iso3_missing_count"] = overview_display["iso3_missing_count"].fillna("n/a")

    md_lines = [
        "# Data Quality Scorecard",
        "",
        "## Dataset overview",
    ]
    if overview_display.empty:
        md_lines.append("_No datasets found._")
    else:
        md_lines.extend(
            markdown_table(
                overview_display[
                    [
                        "dataset",
                        "rows",
                        "columns",
                        "year_min",
                        "year_max",
                        "iso3_missing_count",
                        "iso3_missing_pct",
                        "duplicate_rows",
                    ]
                ]
            )
        )

    md_lines.extend(["", "## ISO3 unmatched (by source_type)"])
    if iso3_unmatched_path.exists():
        iso_df = pd.read_csv(iso3_unmatched_path)
        if iso_df.empty:
            md_lines.append("_No unmatched ISO3 records._")
        else:
            md_lines.extend(markdown_table(iso_df))
    else:
        md_lines.append("_No unmatched ISO3 file found._")

    md_lines.extend(["", "## WHO data_quality distribution"])
    if who_quality_path.exists():
        dq_df = pd.read_csv(who_quality_path)
        if dq_df.empty:
            md_lines.append("_No data_quality rows found._")
        else:
            md_lines.extend(markdown_table(dq_df))
    else:
        md_lines.append("_WHO data_quality file not found._")

    md_lines.extend(["", "## Missingness (top 12 columns)"])
    if missingness_df.empty:
        md_lines.append("_No missingness detected._")
    else:
        top_missing = missingness_df.head(12)
        md_lines.extend(markdown_table(top_missing))

    md_lines.extend(
        [
            "",
            "## Files",
            f"- {overview_path.relative_to(REPO_ROOT)}",
            f"- {missingness_path.relative_to(REPO_ROOT)}",
            f"- {who_quality_path.relative_to(REPO_ROOT)}",
            f"- {iso3_unmatched_path.relative_to(REPO_ROOT)}",
        ]
    )

    report_path = REPORT_DIR / "data_quality_scorecard.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote data quality outputs:")
    print(f"- {overview_path}")
    print(f"- {missingness_path}")
    print(f"- {who_quality_path}")
    print(f"- {iso3_unmatched_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
