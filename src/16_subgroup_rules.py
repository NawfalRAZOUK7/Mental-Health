#!/usr/bin/env python3
"""Subgroup discovery + real association rules (with categoricals) on real data.

Two complementary pattern-mining approaches, both targeting *high suicide rate*
(top tercile) on the REAL enriched country data:

1. Subgroup discovery (pysubgroup, beam search): finds interpretable condition
   combinations whose suicide share is far above the population base rate.
2. Association rules (FP-Growth, mlxtend): transactions include terciled covariates
   AND categorical region/income/sex, so rules are non-trivial, e.g.
   'high alcohol + high unemployment => high suicide'.

Outputs (v1/report/): subgroups.csv, assoc_rules_real.csv, patterns.md
"""
from __future__ import annotations

import warnings

import pandas as pd

try:
    import pysubgroup as ps
    from mlxtend.frequent_patterns import association_rules, fpgrowth
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

from advanced_common import TARGET, load_enriched
from project_paths import REPORT_DIR, VERSION, ensure_dirs

warnings.filterwarnings("ignore")


def tercile(s: pd.Series) -> pd.Series:
    return pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")


def main() -> None:
    ensure_dirs()
    df, num, cat = load_enriched()
    df = df.copy()
    thr = df[TARGET].quantile(2 / 3)
    df["high_suicide"] = df[TARGET] >= thr
    base_rate = float(df["high_suicide"].mean())

    # Keep covariates that are reasonably complete for interpretable patterns.
    keep_num = [c for c in num if df[c].notna().mean() > 0.7]
    for c in keep_num:
        df[c] = df[c].fillna(df[c].median())

    # ---- 1) Subgroup discovery ----
    sg_df = df[[*keep_num, *cat, "high_suicide"]].copy()
    target = ps.BinaryTarget("high_suicide", True)
    searchspace = ps.create_selectors(sg_df, ignore=["high_suicide"])
    task = ps.SubgroupDiscoveryTask(
        sg_df, target, searchspace, result_set_size=12, depth=2, qf=ps.WRAccQF()
    )
    result = ps.BeamSearch().execute(task)
    sg = result.to_dataframe()
    # Normalize columns across pysubgroup versions.
    cov_col = "size_sg" if "size_sg" in sg.columns else ("size" if "size" in sg.columns else None)
    pos_col = "positives_sg" if "positives_sg" in sg.columns else None
    sg_out = pd.DataFrame({
        "subgroup": sg["subgroup"].astype(str),
        "quality": sg["quality"].round(4),
    })
    if cov_col:
        sg_out["n_countries"] = sg[cov_col]
    if pos_col:
        sg_out["share_high"] = (sg[pos_col] / sg[cov_col]).round(3)
    sg_out.to_csv(REPORT_DIR / "subgroups.csv", index=False)

    # ---- 2) Association rules with categoricals (FP-Growth) ----
    items = pd.DataFrame(index=df.index)
    for c in keep_num:
        b = tercile(df[c])
        for level in ["low", "high"]:  # skip 'mid' (least informative)
            items[f"{c}={level}"] = (b == level).astype(bool)
    for c in cat:
        for val in df[c].unique():
            items[f"{c}={val}"] = (df[c] == val).astype(bool)
    items["suicide=high"] = df["high_suicide"].astype(bool)

    freq = fpgrowth(items, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"suicide=high"}))]
    rules = rules[rules["lift"] > 1.2].sort_values("lift", ascending=False)
    rules_out = rules.assign(
        antecedents=rules["antecedents"].apply(lambda s: " + ".join(sorted(s))),
        consequents=rules["consequents"].apply(lambda s: ", ".join(sorted(s))),
    )[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
    rules_out.to_csv(REPORT_DIR / "assoc_rules_real.csv", index=False)

    lines = [
        "# Pattern Mining (real data): subgroups + association rules",
        "",
        f"Target: high suicide rate (top tercile). Base rate: {base_rate:.0%} of countries.",
        "",
        "## Top subgroups (share of high-suicide countries far above base rate)",
    ]
    for r in sg_out.head(5).itertuples():
        share = f", share {r.share_high:.0%}" if "share_high" in sg_out.columns else ""
        n = f", n={int(r.n_countries)}" if "n_countries" in sg_out.columns else ""
        lines.append(f"- {r.subgroup} (quality {r.quality:.3f}{n}{share})")
    lines += ["", "## Top association rules => high suicide (lift > 1.2)"]
    if len(rules_out):
        for r in rules_out.head(6).itertuples():
            lines.append(f"- {r.antecedents} => high suicide (conf {r.confidence:.2f}, lift {r.lift:.2f})")
    else:
        lines.append("- (No rules above thresholds; try lowering min_support.)")
    lines += ["", "## Outputs", "- subgroups.csv, assoc_rules_real.csv"]
    (REPORT_DIR / "patterns.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] subgroups={len(sg_out)}, rules->high_suicide={len(rules_out)}")


if __name__ == "__main__":
    main()
