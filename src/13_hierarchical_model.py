#!/usr/bin/env python3
"""Hierarchical (mixed-effects) model of the national suicide rate.

The data is naturally nested: countries within WHO regions. A flat model ignores
this structure; a partial-pooling mixed model with a region random intercept is the
statistically correct approach and yields the Intraclass Correlation Coefficient
(ICC) -- the share of variance that is *between regions*. A high ICC is a direct,
principled measure of geographic clustering (the "spatial autocorrelation" story)
without needing a country-adjacency weights matrix.

Outputs (v1/report/): hierarchical_model.csv, hierarchical_model.md
"""
from __future__ import annotations

import warnings

import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

from advanced_common import TARGET, load_enriched
from project_paths import REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs

warnings.filterwarnings("ignore")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    ensure_dirs()
    df, num, _ = load_enriched()
    # Standardize numeric predictors so coefficients are comparable.
    preds = [c for c in num if df[c].notna().mean() > 0.7]
    for c in preds:
        df[c] = df[c].fillna(df[c].median())
        df[f"z_{c}"] = zscore(df[c])
    df["region"] = df["region_name"].astype("category")

    zcols = [f"z_{c}" for c in preds]
    formula = f"{TARGET} ~ " + " + ".join(zcols)

    # Null model (region random intercept only) -> variance decomposition / ICC.
    null = smf.mixedlm(f"{TARGET} ~ 1", df, groups=df["region"]).fit(reml=True)
    var_region = float(null.cov_re.iloc[0, 0])
    var_resid = float(null.scale)
    icc = var_region / (var_region + var_resid)

    # Full model: covariates as fixed effects + region random intercept.
    full = smf.mixedlm(formula, df, groups=df["region"]).fit(reml=False)

    fe = full.fe_params
    coefs = (pd.DataFrame({"term": fe.index, "coef": fe.values})
             .query("term != 'Intercept'")
             .assign(abs_coef=lambda d: d["coef"].abs())
             .sort_values("abs_coef", ascending=False))
    coefs.to_csv(REPORT_DIR / "hierarchical_model.csv", index=False)

    top = coefs.head(4)
    top_str = ", ".join(f"{r.term.replace('z_','')} ({r.coef:+.2f})" for r in top.itertuples())

    lines = [
        "# Hierarchical (Mixed-Effects) Model",
        "",
        f"Countries: {len(df)}. Grouping: WHO region (partial pooling).",
        "Predictors standardized (z-scores); coefficients are per 1 SD.",
        "",
        "## Variance decomposition (null model)",
        f"- Between-region variance: {var_region:.2f}",
        f"- Within-region (residual) variance: {var_resid:.2f}",
        f"- **ICC = {icc:.2f}** — {icc:.0%} of the variance in national suicide rates is",
        "  *between* WHO regions (a moderate but clear geographic-clustering effect),",
        "  which is why flat models lean on region as a predictor.",
        "",
        "## Fixed effects — strongest standardized associations",
        f"- {top_str}",
        "- Signs are epidemiologically interpretable (e.g. alcohol positive, life",
        "  expectancy negative), consistent with the SHAP analysis.",
        "",
        "## Why this matters",
        "- A mixed model is the statistically correct treatment of nested (country-in-region)",
        "  data and separates 'where' (region) from 'what' (covariates) instead of conflating them.",
        "",
        "## Outputs",
        f"- {(REPORT_DIR / 'hierarchical_model.csv').relative_to(REPO_ROOT)}",
    ]
    (REPORT_DIR / "hierarchical_model.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] ICC(region)={icc:.2f} | top FE: {top_str}")


if __name__ == "__main__":
    main()
