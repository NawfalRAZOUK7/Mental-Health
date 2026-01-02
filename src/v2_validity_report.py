#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


FEATURE_MAP = {
    "suicide_rate": "age_standardized_suicide_rate_2021",
    "depression_dalys_rate": "gbd_depression_dalys_rate_both",
    "addiction_death_rate": "gbd_addiction_death_rate_both",
    "selfharm_death_rate": "gbd_selfharm_death_rate_both",
}


def summarize(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p10": float(series.quantile(0.1)),
        "p90": float(series.quantile(0.9)),
    }


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    v1_path = REPO_ROOT / "v1" / "data_clean" / "merged_ml_country.csv"
    v2_path = DATA_CLEAN / "synth_country_year.csv"
    if not v1_path.exists() or not v2_path.exists():
        raise SystemExit("Missing v1 or v2 inputs. Build v1 and v2 outputs first.")

    v1 = pd.read_csv(v1_path)
    for col in [
        "age_standardized_suicide_rate_2021",
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_selfharm_death_rate_male",
        "gbd_selfharm_death_rate_female",
    ]:
        v1[col] = pd.to_numeric(v1[col], errors="coerce")
    v1["gbd_selfharm_death_rate_both"] = v1[
        ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
    ].mean(axis=1)

    v1 = v1.groupby("iso3", as_index=False)[list(FEATURE_MAP.values())].mean()

    v2 = pd.read_csv(v2_path)
    v2 = v2[(v2["sex_name"] == "Both") & (v2["year"] == 2021)].copy()
    for col in FEATURE_MAP.keys():
        v2[col] = pd.to_numeric(v2[col], errors="coerce")

    rows = []
    for v2_feature, v1_feature in FEATURE_MAP.items():
        v1_stats = summarize(v1[v1_feature])
        v2_stats = summarize(v2[v2_feature])
        row = {
            "feature": v2_feature,
            "v1_mean": v1_stats["mean"],
            "v1_median": v1_stats["median"],
            "v1_p10": v1_stats["p10"],
            "v1_p90": v1_stats["p90"],
            "v2_mean": v2_stats["mean"],
            "v2_median": v2_stats["median"],
            "v2_p10": v2_stats["p10"],
            "v2_p90": v2_stats["p90"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    test_rows = []
    if stats is not None:
        for v2_feature, v1_feature in FEATURE_MAP.items():
            v1_vals = v1[v1_feature].dropna()
            v2_vals = v2[v2_feature].dropna()
            if v1_vals.empty or v2_vals.empty:
                ks_stat = float("nan")
                ks_pvalue = float("nan")
                w_dist = float("nan")
            else:
                ks = stats.ks_2samp(v1_vals, v2_vals, alternative="two-sided", mode="auto")
                ks_stat = float(ks.statistic)
                ks_pvalue = float(ks.pvalue)
                w_dist = float(stats.wasserstein_distance(v1_vals, v2_vals))
            test_rows.append(
                {
                    "feature": v2_feature,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "wasserstein": w_dist,
                }
            )
    report_path = REPORT_DIR / "v2_validity_report.md"
    lines = [
        "# v2 Synthetic Validity Report",
        "",
        "Comparison of v1 (real) vs v2 (synthetic) distributions for 2021 (Both sexes).",
        "",
        "| feature | v1_mean | v1_median | v1_p10 | v1_p90 | v2_mean | v2_median | v2_p10 | v2_p90 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['feature']} | {row['v1_mean']:.2f} | {row['v1_median']:.2f} | "
            f"{row['v1_p10']:.2f} | {row['v1_p90']:.2f} | {row['v2_mean']:.2f} | "
            f"{row['v2_median']:.2f} | {row['v2_p10']:.2f} | {row['v2_p90']:.2f} |"
        )

    lines.extend(["", "## Distribution Tests"])
    if stats is None:
        lines.append("scipy is not installed; KS/Wasserstein tests skipped.")
    else:
        tests = pd.DataFrame(test_rows)
        lines.extend(
            [
                "",
                "| feature | ks_stat | ks_pvalue | wasserstein |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in tests.iterrows():
            lines.append(
                f"| {row['feature']} | {row['ks_stat']:.4f} | {row['ks_pvalue']:.4f} | {row['wasserstein']:.4f} |"
            )
    note_lines = [
        "",
        "Notes:",
        "- v1 values are averaged across age groups per country.",
        "- v2 values are synthetic and intended for demonstration only.",
        "- Synthetic rates may differ in scale from v1; interpret patterns over magnitudes.",
    ]
    if stats is None:
        note_lines.append("- KS/Wasserstein tests skipped (scipy not installed).")
    else:
        note_lines.append("- KS/Wasserstein tests included via scipy.")
    lines.extend(note_lines)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote validity report to {report_path}")


if __name__ == "__main__":
    main()
