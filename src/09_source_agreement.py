#!/usr/bin/env python3
"""Source-agreement analysis: WHO age-standardized suicide rate vs IHME self-harm death rate.

Motivation
----------
In the GBD taxonomy, the cause "Self-harm" corresponds almost exactly to suicide.
The v1 ML baseline therefore risks a near-tautology if the IHME self-harm death rate
is used to predict the WHO suicide rate. This script makes that relationship explicit
and turns it into a legitimate finding: *how consistently do two independent global
health bodies (WHO and IHME) measure the same phenomenon?*

Outputs (v1/report/)
--------------------
- source_agreement_metrics.csv : correlation + Bland-Altman statistics
- source_agreement.md          : short human-readable summary
report_latex/figures/
- fig_v1_source_agreement.png  : scatter (WHO vs IHME) + Bland-Altman plot
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:
    raise SystemExit("scipy is required. pip install -r requirements.txt") from exc

import theme
from project_paths import DATA_CLEAN, REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs


def load_country_level() -> pd.DataFrame:
    path = DATA_CLEAN / "merged_ml_country.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 04_merge_ml.py first.")
    df = pd.read_csv(path)
    for col in [
        "age_standardized_suicide_rate_2021",
        "gbd_selfharm_death_rate_male",
        "gbd_selfharm_death_rate_female",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["selfharm_both"] = df[
        ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
    ].mean(axis=1)
    g = (
        df.groupby(["iso3", "location_name", "region_name"], as_index=False)
        .agg(
            who_suicide=("age_standardized_suicide_rate_2021", "mean"),
            ihme_selfharm=("selfharm_both", "mean"),
        )
        .dropna(subset=["who_suicide", "ihme_selfharm"])
    )
    return g


def main() -> None:
    ensure_dirs()
    theme.apply_matplotlib()
    g = load_country_level()
    who, ihme = g["who_suicide"], g["ihme_selfharm"]
    diff = who - ihme
    mean_of_two = (who + ihme) / 2

    pear = stats.pearsonr(who, ihme)
    spear = stats.spearmanr(who, ihme)
    loa_low = diff.mean() - 1.96 * diff.std()
    loa_high = diff.mean() + 1.96 * diff.std()

    metrics = pd.DataFrame(
        [
            {"metric": "n_countries", "value": len(g)},
            {"metric": "pearson_r", "value": round(float(pear.statistic), 4)},
            {"metric": "pearson_r2", "value": round(float(pear.statistic) ** 2, 4)},
            {"metric": "spearman_rho", "value": round(float(spear.statistic), 4)},
            {"metric": "mean_diff_who_minus_ihme", "value": round(float(diff.mean()), 3)},
            {"metric": "sd_diff", "value": round(float(diff.std()), 3)},
            {"metric": "loa_low", "value": round(float(loa_low), 3)},
            {"metric": "loa_high", "value": round(float(loa_high), 3)},
            {"metric": "median_abs_diff", "value": round(float(diff.abs().median()), 3)},
        ]
    )
    metrics_path = REPORT_DIR / "source_agreement_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # ---- Figure: scatter + Bland-Altman ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = max(who.max(), ihme.max()) * 1.05
    axes[0].scatter(ihme, who, s=18, alpha=0.7, color="#1f6f8b", edgecolor="white", linewidth=0.4)
    axes[0].plot([0, lim], [0, lim], "--", color="#b0453c", linewidth=1, label="y = x")
    axes[0].set_xlabel("IHME self-harm death rate (per 100k)")
    axes[0].set_ylabel("WHO age-standardized suicide rate (per 100k)")
    axes[0].set_title(f"Cross-source agreement\nPearson r = {pear.statistic:.3f}  (R² = {pear.statistic**2:.3f})")
    axes[0].legend(frameon=False)

    axes[1].scatter(mean_of_two, diff, s=18, alpha=0.7, color="#1f6f8b", edgecolor="white", linewidth=0.4)
    axes[1].axhline(diff.mean(), color="#333333", linewidth=1, label=f"mean {diff.mean():.2f}")
    axes[1].axhline(loa_high, color="#b0453c", linestyle="--", linewidth=1, label="±1.96 SD")
    axes[1].axhline(loa_low, color="#b0453c", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Mean of the two measures (per 100k)")
    axes[1].set_ylabel("WHO − IHME (per 100k)")
    axes[1].set_title("Bland–Altman: measurement difference")
    axes[1].legend(frameon=False)
    fig.tight_layout()

    fig_dir = REPO_ROOT / "report_latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig_v1_source_agreement.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    summary = [
        "# Source Agreement: WHO vs IHME",
        "",
        "Two independent global bodies measure suicide/self-harm. This checks how closely",
        "the WHO age-standardized suicide rate and the IHME self-harm death rate agree,",
        "and documents why self-harm must not be used as a 'predictor' of suicide.",
        "",
        f"- Countries compared: {len(g)}",
        f"- Pearson r: {pear.statistic:.3f} (R² = {pear.statistic**2:.3f})",
        f"- Spearman rho: {spear.statistic:.3f}",
        f"- Mean difference (WHO − IHME): {diff.mean():.2f} per 100k",
        f"- Limits of agreement: [{loa_low:.2f}, {loa_high:.2f}] per 100k",
        f"- Median absolute difference: {diff.abs().median():.2f} per 100k",
        "",
        "**Interpretation.** The two sources agree strongly and with little systematic bias,",
        "confirming they measure essentially the same phenomenon. Consequently, the enriched",
        "ML model (see 11_ml_enriched.py) excludes self-harm as a feature and predicts the",
        "suicide rate from *independent* drivers (depression burden, addiction burden,",
        "geography, and socioeconomic covariates).",
        "",
        "## Outputs",
        f"- {metrics_path.relative_to(REPO_ROOT)}",
        f"- {fig_path.relative_to(REPO_ROOT)}",
    ]
    (REPORT_DIR / "source_agreement.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"[{VERSION}] Source agreement: r={pear.statistic:.3f}, R2={pear.statistic**2:.3f}")
    print(f"[{VERSION}] Wrote {metrics_path}")
    print(f"[{VERSION}] Wrote {fig_path}")


if __name__ == "__main__":
    main()
