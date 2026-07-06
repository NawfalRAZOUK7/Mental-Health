#!/usr/bin/env python3
"""Country similarity network + community detection on real enriched data.

Builds a k-nearest-neighbour graph over countries (edges = similar mental-health /
socioeconomic profiles), detects communities via greedy modularity, and computes
centrality. Answers 'which countries cluster together, and which are bridges?'
on REAL data (unlike the synthetic v2 graph demo).

Outputs (v1/report/): network_nodes.csv, network_edges.csv, network.md
Figures (report_latex/figures/): fig_v1_country_network.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import networkx as nx
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

import theme
from advanced_common import TARGET, load_enriched
from project_paths import REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs

RNG = 42
K = 4
# On-brand 10-color palette for communities (cohesive with the design system).
COMMUNITY_COLORS = [
    "#1f6f8b", "#b0453c", "#d99a2b", "#2f8f6b", "#7c6f9e",
    "#8f877f", "#185a72", "#d98b7a", "#b8801f", "#6fae91",
]


def main() -> None:
    ensure_dirs()
    theme.apply_matplotlib()
    df, num, _ = load_enriched()
    X = SimpleImputer(strategy="median").fit_transform(df[num])
    X = StandardScaler().fit_transform(X)

    nn = NearestNeighbors(n_neighbors=K + 1).fit(X)
    dist, idx = nn.kneighbors(X)

    G = nx.Graph()
    for i, iso in enumerate(df["iso3"]):
        G.add_node(iso, name=df["location_name"].iloc[i], suicide=float(df[TARGET].iloc[i]),
                   region=df["region_name"].iloc[i])
    edges = []
    for i in range(len(df)):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            a, b = df["iso3"].iloc[i], df["iso3"].iloc[j]
            w = 1.0 / (1.0 + float(d))
            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=w)
                edges.append({"source": a, "target": b, "weight": round(w, 4)})

    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    comm_map = {n: c for c, comm in enumerate(communities) for n in comm}
    modularity = nx.community.modularity(G, communities, weight="weight")
    btw = nx.betweenness_centrality(G, weight="weight")
    deg = dict(G.degree())

    nodes = pd.DataFrame([
        {"iso3": n, "name": G.nodes[n]["name"], "region": G.nodes[n]["region"],
         "suicide_rate": G.nodes[n]["suicide"], "community": comm_map[n],
         "degree": deg[n], "betweenness": round(btw[n], 4)}
        for n in G.nodes
    ]).sort_values("betweenness", ascending=False)
    nodes.to_csv(REPORT_DIR / "network_nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(REPORT_DIR / "network_edges.csv", index=False)

    # Figure
    pos = nx.spring_layout(G, seed=RNG, weight="weight", k=0.35)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    node_colors = [COMMUNITY_COLORS[comm_map[n] % len(COMMUNITY_COLORS)] for n in G.nodes]
    node_sizes = [40 + 12 * G.nodes[n]["suicide"] for n in G.nodes]
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, linewidths=0.3, edgecolors="white", ax=ax)
    # Label the most central bridge countries.
    top_bridges = nodes.head(12)["iso3"].tolist()
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_bridges}, font_size=8, ax=ax)
    ax.set_title(f"Country similarity network — {len(communities)} communities "
                 f"(modularity {modularity:.2f}); node size = suicide rate, labels = top bridges")
    ax.axis("off")
    fig.tight_layout()
    fig_dir = REPO_ROOT / "report_latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_v1_country_network.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Country Similarity Network (real data)",
        "",
        f"- Nodes: {G.number_of_nodes()} countries; Edges: {G.number_of_edges()} (k-NN, k={K}).",
        f"- Communities (greedy modularity): {len(communities)}; modularity = {modularity:.2f}.",
        f"- Sizes: {[len(c) for c in communities]}",
        "",
        "## Most central 'bridge' countries (betweenness)",
    ]
    for r in nodes.head(6).itertuples():
        lines.append(f"- {r.name} ({r.iso3}): betweenness {r.betweenness:.3f}, community {r.community}")
    lines += ["", "## Outputs", "- network_nodes.csv, network_edges.csv",
              "- report_latex/figures/fig_v1_country_network.png"]
    (REPORT_DIR / "network.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] network: {len(communities)} communities, modularity={modularity:.2f}, "
          f"edges={G.number_of_edges()}")


if __name__ == "__main__":
    main()
