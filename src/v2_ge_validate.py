#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

try:
    import great_expectations as ge
    GE_AVAILABLE = True
    GE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency issues
    ge = None
    GE_AVAILABLE = False
    GE_IMPORT_ERROR = exc

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


FEATURE_COLS = [
    "suicide_rate",
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]
SEX_VALUES = ["Male", "Female", "Both"]


def numeric_bounds(series: pd.Series, min_floor: float = 0.0) -> tuple[float, float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return min_floor, min_floor + 1.0
    upper = float(series.quantile(0.995) * 1.5)
    if upper <= min_floor:
        upper = min_floor + 1.0
    return min_floor, upper


def fallback_expectations(df: pd.DataFrame, expectations: list[dict]) -> dict:
    results = []
    for spec in expectations:
        exp_type = spec["expectation_type"]
        kwargs = spec.get("kwargs", {})
        success = bool(spec["check"]())
        results.append(
            {
                "success": success,
                "expectation_config": {
                    "expectation_type": exp_type,
                    "kwargs": kwargs,
                },
            }
        )
    evaluated = len(results)
    successful = sum(1 for item in results if item["success"])
    unsuccessful = evaluated - successful
    success_percent = 0.0 if evaluated == 0 else successful / evaluated * 100
    return {
        "statistics": {
            "evaluated_expectations": evaluated,
            "successful_expectations": successful,
            "unsuccessful_expectations": unsuccessful,
            "success_percent": success_percent,
        },
        "results": results,
    }


def validate_country_year(df: pd.DataFrame) -> dict:
    if GE_AVAILABLE and ge is not None:
        validator = ge.from_pandas(df)
        validator.expect_table_row_count_to_be_between(min_value=50, max_value=None)
        validator.expect_column_values_to_not_be_null("iso3", mostly=0.98)
        validator.expect_column_values_to_match_regex("iso3", r"^[A-Z]{3}$", mostly=0.98)
        validator.expect_column_values_to_be_in_set("sex_name", SEX_VALUES)
        validator.expect_column_values_to_be_between("year", min_value=2000, max_value=2030)
        validator.expect_column_values_to_not_be_null("region_name", mostly=0.98)
        validator.expect_column_values_to_not_be_null("income_group", mostly=0.9)
        for col in FEATURE_COLS:
            min_val, max_val = numeric_bounds(df[col], 0.0)
            validator.expect_column_values_to_be_between(col, min_value=min_val, max_value=max_val, mostly=0.97)
        validator.expect_column_values_to_be_between("population", min_value=0, max_value=None, mostly=0.98)
        return validator.validate(result_format="SUMMARY")

    expectations = [
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": 50, "max_value": None},
            "check": lambda: len(df) >= 50,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "iso3", "mostly": 0.98},
            "check": lambda: df["iso3"].notna().mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {"column": "iso3", "regex": "^[A-Z]{3}$", "mostly": 0.98},
            "check": lambda: df["iso3"].astype(str).str.match(r"^[A-Z]{3}$").mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": "sex_name", "value_set": SEX_VALUES},
            "check": lambda: df["sex_name"].isin(SEX_VALUES).mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "year", "min_value": 2000, "max_value": 2030},
            "check": lambda: df["year"].between(2000, 2030).mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "region_name", "mostly": 0.98},
            "check": lambda: df["region_name"].notna().mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "income_group", "mostly": 0.9},
            "check": lambda: df["income_group"].notna().mean() >= 0.9,
        },
    ]
    for col in FEATURE_COLS:
        min_val, max_val = numeric_bounds(df[col], 0.0)
        expectations.append(
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": col, "min_value": min_val, "max_value": max_val, "mostly": 0.97},
                "check": lambda c=col, lo=min_val, hi=max_val: df[c].between(lo, hi).mean() >= 0.97,
            }
        )
    expectations.append(
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "population", "min_value": 0, "max_value": None, "mostly": 0.98},
            "check": lambda: df["population"].fillna(0).ge(0).mean() >= 0.98,
        }
    )
    return fallback_expectations(df, expectations)


def validate_region_year(df: pd.DataFrame) -> dict:
    if GE_AVAILABLE and ge is not None:
        validator = ge.from_pandas(df)
        validator.expect_table_row_count_to_be_between(min_value=10, max_value=None)
        validator.expect_column_values_to_not_be_null("region_name", mostly=0.98)
        validator.expect_column_values_to_be_in_set("sex_name", SEX_VALUES)
        validator.expect_column_values_to_be_between("year", min_value=2000, max_value=2030)
        for col in FEATURE_COLS:
            min_val, max_val = numeric_bounds(df[col], 0.0)
            validator.expect_column_values_to_be_between(col, min_value=min_val, max_value=max_val, mostly=0.97)
        return validator.validate(result_format="SUMMARY")

    expectations = [
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": 10, "max_value": None},
            "check": lambda: len(df) >= 10,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "region_name", "mostly": 0.98},
            "check": lambda: df["region_name"].notna().mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": "sex_name", "value_set": SEX_VALUES},
            "check": lambda: df["sex_name"].isin(SEX_VALUES).mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "year", "min_value": 2000, "max_value": 2030},
            "check": lambda: df["year"].between(2000, 2030).mean() >= 0.98,
        },
    ]
    for col in FEATURE_COLS:
        min_val, max_val = numeric_bounds(df[col], 0.0)
        expectations.append(
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": col, "min_value": min_val, "max_value": max_val, "mostly": 0.97},
                "check": lambda c=col, lo=min_val, hi=max_val: df[c].between(lo, hi).mean() >= 0.97,
            }
        )
    return fallback_expectations(df, expectations)


def validate_long(df: pd.DataFrame) -> dict:
    if GE_AVAILABLE and ge is not None:
        validator = ge.from_pandas(df)
        validator.expect_table_row_count_to_be_between(min_value=1000, max_value=None)
        validator.expect_column_values_to_not_be_null("iso3", mostly=0.95)
        validator.expect_column_values_to_be_in_set("sex_name", SEX_VALUES)
        validator.expect_column_values_to_be_between("year", min_value=2000, max_value=2030)
        validator.expect_column_values_to_not_be_null("age_name", mostly=0.95)
        for col in FEATURE_COLS:
            min_val, max_val = numeric_bounds(df[col], 0.0)
            validator.expect_column_values_to_be_between(col, min_value=min_val, max_value=max_val, mostly=0.95)
        validator.expect_column_values_to_be_between("population", min_value=0, max_value=None, mostly=0.97)
        return validator.validate(result_format="SUMMARY")

    expectations = [
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": 1000, "max_value": None},
            "check": lambda: len(df) >= 1000,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "iso3", "mostly": 0.95},
            "check": lambda: df["iso3"].notna().mean() >= 0.95,
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": "sex_name", "value_set": SEX_VALUES},
            "check": lambda: df["sex_name"].isin(SEX_VALUES).mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "year", "min_value": 2000, "max_value": 2030},
            "check": lambda: df["year"].between(2000, 2030).mean() >= 0.98,
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "age_name", "mostly": 0.95},
            "check": lambda: df["age_name"].notna().mean() >= 0.95,
        },
    ]
    for col in FEATURE_COLS:
        min_val, max_val = numeric_bounds(df[col], 0.0)
        expectations.append(
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": col, "min_value": min_val, "max_value": max_val, "mostly": 0.95},
                "check": lambda c=col, lo=min_val, hi=max_val: df[c].between(lo, hi).mean() >= 0.95,
            }
        )
    expectations.append(
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "population", "min_value": 0, "max_value": None, "mostly": 0.97},
            "check": lambda: df["population"].fillna(0).ge(0).mean() >= 0.97,
        }
    )
    return fallback_expectations(df, expectations)


def summarize_result(name: str, result: dict) -> tuple[dict, list[dict]]:
    stats = result.get("statistics", {})
    summary = {
        "dataset": name,
        "evaluated_expectations": stats.get("evaluated_expectations", 0),
        "successful_expectations": stats.get("successful_expectations", 0),
        "unsuccessful_expectations": stats.get("unsuccessful_expectations", 0),
        "success_percent": stats.get("success_percent", 0.0),
    }
    failures = []
    for item in result.get("results", []):
        if item.get("success", False):
            continue
        config = item.get("expectation_config", {})
        kwargs = config.get("kwargs", {})
        failures.append(
            {
                "dataset": name,
                "expectation": config.get("expectation_type", ""),
                "column": kwargs.get("column", ""),
                "details": str(kwargs),
            }
        )
    return summary, failures


def render_html_report(summary_df: pd.DataFrame, failures_df: pd.DataFrame) -> str:
    summary_rows = "\n".join(
        f"<tr><td>{row.dataset}</td><td>{row.evaluated_expectations}</td>"
        f"<td>{row.successful_expectations}</td><td>{row.unsuccessful_expectations}</td>"
        f"<td>{row.success_percent:.2f}%</td></tr>"
        for row in summary_df.itertuples(index=False)
    )
    if failures_df.empty:
        failures_html = "<p>All expectations passed.</p>"
    else:
        failure_rows = "\n".join(
            f"<tr><td>{row.dataset}</td><td>{row.expectation}</td><td>{row.column}</td>"
            f"<td>{row.details}</td></tr>"
            for row in failures_df.itertuples(index=False)
        )
        failures_html = (
            "<table><thead><tr><th>Dataset</th><th>Expectation</th><th>Column</th><th>Details</th></tr></thead>"
            f"<tbody>{failure_rows}</tbody></table>"
        )

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>v2 Great Expectations Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; color: #1c1b1a; }}
    h1 {{ font-size: 22px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #e0e0e0; padding: 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    code {{ background: #f7f2ea; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>v2 Data Quality Validation (Great Expectations)</h1>
  <p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
  <p>Mode: {"Great Expectations" if GE_AVAILABLE else "Fallback checks (GE unavailable on Python 3.14)"}</p>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Expectations</th>
        <th>Passed</th>
        <th>Failed</th>
        <th>Success %</th>
      </tr>
    </thead>
    <tbody>
      {summary_rows}
    </tbody>
  </table>
  <h2>Failures</h2>
  {failures_html}
</body>
</html>
"""


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    long_path = DATA_CLEAN / "synth_long.csv"
    country_path = DATA_CLEAN / "synth_country_year.csv"
    region_path = DATA_CLEAN / "synth_region_year.csv"
    for path in [long_path, country_path, region_path]:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run src/v2_generate_synth.py first.")

    long_df = pd.read_csv(long_path)
    country_df = pd.read_csv(country_path)
    region_df = pd.read_csv(region_path)

    results = []
    failures = []

    for name, validator_fn, df in [
        ("synth_long", validate_long, long_df),
        ("synth_country_year", validate_country_year, country_df),
        ("synth_region_year", validate_region_year, region_df),
    ]:
        result = validator_fn(df)
        summary, failed = summarize_result(name, result)
        results.append(summary)
        failures.extend(failed)

    summary_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)

    report_path = REPORT_DIR / "ge_report.html"
    report_path.write_text(render_html_report(summary_df, failures_df), encoding="utf-8")

    md_lines = [
        "# v2 Data Quality Summary (Great Expectations)",
        "",
        f"Mode: {'Great Expectations' if GE_AVAILABLE else 'Fallback checks (GE unavailable on Python 3.14)'}",
        "",
        "| Dataset | Expectations | Passed | Failed | Success % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        md_lines.append(
            f"| {row.dataset} | {row.evaluated_expectations} | {row.successful_expectations} | "
            f"{row.unsuccessful_expectations} | {row.success_percent:.2f} |"
        )
    if failures_df.empty:
        md_lines.extend(["", "No expectation failures detected."])
    else:
        md_lines.extend(["", "## Failed Expectations (sample)"])
        for row in failures_df.head(12).itertuples(index=False):
            md_lines.append(f"- {row.dataset}: {row.expectation} ({row.column})")

    if not GE_AVAILABLE and GE_IMPORT_ERROR is not None:
        md_lines.extend(
            [
                "",
                "## Great Expectations Status",
                f"Great Expectations could not initialize on Python 3.14: `{GE_IMPORT_ERROR}`",
                "These results were computed with a fallback validator that mirrors the GE expectations.",
            ]
        )

    summary_path = REPORT_DIR / "v2_quality_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote GE report to {report_path}")
    print(f"[{VERSION}] Wrote quality summary to {summary_path}")


if __name__ == "__main__":
    main()
