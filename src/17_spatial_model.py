#!/usr/bin/env python3
"""Spatial autocorrelation of suicide rates on the country-similarity graph.

We have no free geographic-adjacency dataset, so "neighbours" are defined in
*feature space*: the k-NN country-similarity network built by
15_country_network.py (edges = similar mental-health / socioeconomic profiles).
This is a network-autocorrelation analysis, not a geographic one — stated
plainly so it is not oversold.

We ask two questions:
  1. Do countries with similar profiles have similar suicide rates beyond what
     the leakage-free model already captures?  -> Moran's I on the model
     *residuals* (actual - predicted from 12_ml_advanced.py).
  2. Where are the local clusters?  -> Local Moran's I (LISA) quadrants.

Optionally fits a spatial-lag regression (spreg) and compares it to OLS.

Opt-in: only runs when MHV_SPATIAL=1 (see scripts/run_advanced.py).

Outputs (v1/report/): spatial_model.csv, spatial_model.json, spatial_model.md
Figures (report_latex/figures/): fig_v1_spatial_lisa.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import W
except ImportError as exc:
    raise SystemExit(
        "Spatial model needs PySAL. Run: pip install libpysal esda spreg"
    ) from exc

import theme
from advanced_common import TARGET, load_enriched
from project_paths import REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs

RNG = 42
PERMUTATIONS = 999
SIG = 0.05
# LISA quadrant labels (High-High, Low-Low, Low-High, High-Low).
QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
QUAD_COLOR = {
    "High-High": "#b0453c",  # hot cluster
    "Low-Low": "#1f6f8b",    # cold cluster
    "Low-High": "#7fb5c9",   # spatial outlier
    "High-Low": "#d99a2b",   # spatial outlier
    "ns": "#c9c2b8",         # not significant
}


def _load_residuals() -> pd.DataFrame:
    """Return per-country actual, predicted, and residual from the leakage-free model."""
    path = REPORT_DIR / "ml_advanced_conformal.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run 12_ml_advanced.py first (produces conformal predictions)."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"suicide_rate": "actual"})
    df["residual"] = df["actual"] - df["pred"]
    return df[["iso3", "location_name", "actual", "pred", "residual"]].dropna(subset=["residual"])


def _build_weights(ids: list[str]) -> W:
    """Build a row-standardized spatial-weights matrix from the k-NN edge list."""
    edges_path = REPORT_DIR / "network_edges.csv"
    if not edges_path.exists():
        raise SystemExit(
            f"Missing {edges_path}. Run 15_country_network.py first (builds the graph)."
        )
    edges = pd.read_csv(edges_path)
    idset = set(ids)
    neighbors: dict[str, list[str]] = {i: [] for i in ids}
    weights: dict[str, list[float]] = {i: [] for i in ids}
    for src, tgt, wgt in edges[["source", "target", "weight"]].itertuples(index=False):
        if src in idset and tgt in idset:
            # symmetrize: the similarity graph is undirected
            for a, b in ((src, tgt), (tgt, src)):
                if b not in neighbors[a]:
                    neighbors[a].append(b)
                    weights[a].append(float(wgt))
    # drop isolates (no neighbours after intersection) — PySAL warns on them
    kept = [i for i in ids if neighbors[i]]
    neighbors = {i: neighbors[i] for i in kept}
    weights = {i: weights[i] for i in kept}
    w = W(neighbors, weights, id_order=kept, silence_warnings=True)
    w.transform = "r"
    return w


def _moran(y: np.ndarray, w: W, label: str) -> dict:
    mi = Moran(y, w, permutations=PERMUTATIONS)
    print(f"[{VERSION}] Moran's I ({label}): I={mi.I:.3f}, E[I]={mi.EI:.3f}, "
          f"p(sim)={mi.p_sim:.3f}")
    return {
        "variable": label,
        "morans_I": round(float(mi.I), 4),
        "expected_I": round(float(mi.EI), 4),
        "z_sim": round(float(mi.z_sim), 4),
        "p_sim": round(float(mi.p_sim), 4),
        "significant": bool(mi.p_sim < SIG),
        "interpretation": _interpret(mi.I, mi.EI, mi.p_sim, label),
    }


def _interpret(moran_i: float, expected_i: float, p: float, label: str) -> str:
    if p >= SIG:
        return (f"No significant network autocorrelation in {label} "
                f"(p={p:.3f}) — similar-profile countries are not more alike than chance.")
    if moran_i > expected_i:
        return (f"Significant positive autocorrelation in {label} "
                f"(I={moran_i:.3f}, p={p:.3f}): countries with similar profiles have "
                f"similar values — they cluster.")
    return (f"Significant negative autocorrelation in {label} "
            f"(I={moran_i:.3f}, p={p:.3f}): similar-profile countries tend to differ "
            f"(a checkerboard pattern).")


def _lisa(y: np.ndarray, w: W, ids: list[str]) -> pd.DataFrame:
    lm = Moran_Local(y, w, permutations=PERMUTATIONS, seed=RNG)
    quad = [QUAD.get(int(q), "ns") for q in lm.q]
    sig = lm.p_sim < SIG
    label = [q if s else "ns" for q, s in zip(quad, sig)]
    return pd.DataFrame({
        "iso3": ids,
        "lisa_quadrant": label,
        "lisa_p": np.round(lm.p_sim, 4),
        "local_I": np.round(lm.Is, 4),
    })


def _spatial_regression(w: W, ids: list[str]) -> dict | None:
    """Optional OLS vs spatial-lag comparison (skipped gracefully if spreg is unavailable)."""
    try:
        from spreg import OLS, GM_Lag
    except ImportError:
        print(f"[{VERSION}] spreg not available — skipping spatial regression.")
        return None
    enriched, num_features, _ = load_enriched()
    enriched = enriched[enriched["iso3"].isin(ids)].copy()
    enriched = enriched.dropna(subset=[TARGET, *num_features])
    common = [i for i in ids if i in set(enriched["iso3"])]
    if len(common) < 30:
        print(f"[{VERSION}] too few countries with covariates ({len(common)}) — skipping regression.")
        return None
    # rebuild weights on the common subset so W matches X row order
    wsub = _build_weights(common)
    order = wsub.id_order
    enriched = enriched.set_index("iso3").loc[order]
    y = enriched[[TARGET]].to_numpy(dtype=float)
    X = enriched[num_features].to_numpy(dtype=float)
    names = list(num_features)
    ols = OLS(y, X, name_x=names, name_y=TARGET, name_ds="countries")
    lag = GM_Lag(y, X, w=wsub, name_x=names, name_y=TARGET, name_ds="countries")
    rho = float(lag.rho) if np.ndim(lag.rho) == 0 else float(np.ravel(lag.rho)[0])
    print(f"[{VERSION}] OLS R2={float(ols.r2):.3f} | spatial-lag pseudo-R2="
          f"{float(lag.pr2):.3f}, rho={rho:.3f}")
    return {
        "n": len(order),
        "ols_r2": round(float(ols.r2), 4),
        "spatial_lag_pseudo_r2": round(float(lag.pr2), 4),
        "spatial_rho": round(rho, 4),
        "note": ("rho is the spatial-lag coefficient on the similarity graph; "
                 "a nonzero rho means a country's suicide rate co-moves with its "
                 "profile-neighbours beyond the covariates."),
    }


def _plot_lisa(df: pd.DataFrame) -> None:
    counts = df["lisa_quadrant"].value_counts()
    order = ["High-High", "Low-Low", "High-Low", "Low-High", "ns"]
    counts = counts.reindex([q for q in order if q in counts.index])
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [QUAD_COLOR.get(q, "#c9c2b8") for q in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Countries")
    ax.set_title("Local Moran's I (LISA) clusters on the similarity graph\n"
                 "residual suicide rate — HH/LL = clusters, HL/LH = spatial outliers")
    fig.tight_layout()
    fig_dir = REPO_ROOT / "report_latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_v1_spatial_lisa.png", dpi=150)
    plt.close(fig)


def main() -> None:
    theme.apply_matplotlib()
    _, report_dir = ensure_dirs()

    res = _load_residuals()
    w = _build_weights(res["iso3"].tolist())
    ids = w.id_order
    res = res.set_index("iso3").loc[ids].reset_index()

    y_rate = res["actual"].to_numpy(dtype=float)
    y_resid = res["residual"].to_numpy(dtype=float)

    moran_rate = _moran(y_rate, w, "suicide_rate")
    moran_resid = _moran(y_resid, w, "model_residual")

    lisa = _lisa(y_resid, w, ids)
    out = res.merge(lisa, on="iso3", how="left")
    out.to_csv(report_dir / "spatial_model.csv", index=False)
    _plot_lisa(out)

    reg = _spatial_regression(w, ids)

    summary = {
        "version": VERSION,
        "n_countries": len(ids),
        "neighbour_definition": "feature-space k-NN similarity graph (not geographic)",
        "permutations": PERMUTATIONS,
        "moran": {"suicide_rate": moran_rate, "model_residual": moran_resid},
        "spatial_regression": reg,
    }
    (report_dir / "spatial_model.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    clusters = out[out["lisa_quadrant"].isin(["High-High", "Low-Low"])]
    hh = out[out["lisa_quadrant"] == "High-High"]
    ll = out[out["lisa_quadrant"] == "Low-Low"]
    lines = [
        "# Spatial autocorrelation (feature-space similarity graph)",
        "",
        "**Neighbours are defined in feature space** (the k-NN country-similarity "
        "network), not by geography — this is network autocorrelation.",
        "",
        f"- Countries analysed: {len(ids)}; permutations: {PERMUTATIONS}.",
        f"- Moran's I (suicide rate): {moran_rate['morans_I']} "
        f"(p={moran_rate['p_sim']}). {moran_rate['interpretation']}",
        f"- Moran's I (model residual): {moran_resid['morans_I']} "
        f"(p={moran_resid['p_sim']}). {moran_resid['interpretation']}",
        f"- Significant LISA clusters: {len(clusters)} "
        f"(High-High: {len(hh)}, Low-Low: {len(ll)}).",
    ]
    if reg:
        lines += [
            f"- Spatial-lag regression: OLS R²={reg['ols_r2']} vs "
            f"pseudo-R²={reg['spatial_lag_pseudo_r2']} (rho={reg['spatial_rho']}).",
        ]
    if len(hh):
        lines += ["", "## High-High clusters (high residual, high-residual neighbours)"]
        lines += [f"- {r.location_name} ({r.iso3}): residual {r.residual:+.1f}"
                  for r in hh.itertuples()]
    if len(ll):
        lines += ["", "## Low-Low clusters (low residual, low-residual neighbours)"]
        lines += [f"- {r.location_name} ({r.iso3}): residual {r.residual:+.1f}"
                  for r in ll.itertuples()]
    lines += ["", "## Outputs",
              "- spatial_model.csv (per-country residual + LISA quadrant)",
              "- spatial_model.json (Moran's I + regression summary)",
              "- report_latex/figures/fig_v1_spatial_lisa.png"]
    (report_dir / "spatial_model.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] spatial: {len(clusters)} significant LISA clusters "
          f"(HH={len(hh)}, LL={len(ll)}).")


if __name__ == "__main__":
    main()
