#!/usr/bin/env python3
"""UMAP embedding of countries on real enriched features.

Projects each country into 2-D from its standardized suicide/epidemiological/
socioeconomic profile, then colors by KMeans cluster and by suicide rate.
A compact, high-signal 'map of countries by mental-health profile'.

Outputs (v1/report/): umap_embedding.csv
Figures (report_latex/figures/): fig_v1_umap_countries.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

import theme
from advanced_common import TARGET, load_enriched
from project_paths import REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs

RNG = 42


def main() -> None:
    ensure_dirs()
    theme.apply_matplotlib()
    df, num, _ = load_enriched()
    X = SimpleImputer(strategy="median").fit_transform(df[num])
    X = StandardScaler().fit_transform(X)

    # Choose k by silhouette.
    best_k, best_s = 4, -1.0
    for k in [3, 4, 5, 6]:
        lab = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best_s:
            best_k, best_s = k, s
    labels = KMeans(n_clusters=best_k, random_state=RNG, n_init=10).fit_predict(X)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RNG)
    emb = reducer.fit_transform(X)

    out = df[["iso3", "location_name", "region_name", TARGET]].copy()
    out["umap_x"], out["umap_y"], out["cluster"] = emb[:, 0], emb[:, 1], labels
    out.to_csv(REPORT_DIR / "umap_embedding.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for c in sorted(set(labels)):
        m = labels == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=26, alpha=0.8, label=f"Cluster {c}")
    axes[0].set_title(f"Countries by profile — KMeans (k={best_k}, silhouette={best_s:.2f})")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    from matplotlib.colors import LinearSegmentedColormap

    mhv_cmap = LinearSegmentedColormap.from_list("mhv", theme.SEQUENTIAL)
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], c=df[TARGET], cmap=mhv_cmap, s=26, alpha=0.9)
    axes[1].set_title("Same map, colored by suicide rate (per 100k)")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=axes[1], label="suicide rate")
    fig.tight_layout()

    fig_dir = REPO_ROOT / "report_latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_v1_umap_countries.png", dpi=150)
    plt.close(fig)

    print(f"[{VERSION}] UMAP done. k={best_k}, silhouette={best_s:.2f}, "
          f"clusters sizes={np.bincount(labels).tolist()}")


if __name__ == "__main__":
    main()
