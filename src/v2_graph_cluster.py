#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "networkx is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


FEATURE_COLS = [
    "suicide_rate",
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]
TOP_K = 5
MIN_SIMILARITY = 0.35
FALLBACK_MIN_SIMILARITY = 0.2
FALLBACK_TOP_K = 10
SEED = 42


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
    df = df.dropna(subset=["iso3"] + FEATURE_COLS)
    df = df.drop_duplicates(subset=["iso3"])
    if df.empty:
        raise SystemExit("No data available for graph clustering.")

    base_cols = [
        "iso3",
        "location_name",
        "region_name",
        "income_group",
        "suicide_rate",
    ] + [col for col in FEATURE_COLS if col != "suicide_rate"]
    base = df[base_cols].copy().reset_index(drop=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(base[FEATURE_COLS])

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)

    def build_edges(min_similarity: float, top_k: int) -> dict[tuple[int, int], float]:
        edges: dict[tuple[int, int], float] = {}
        for i in range(sim.shape[0]):
            order = np.argsort(sim[i])[::-1]
            count = 0
            for j in order:
                if i == j:
                    continue
                score = float(sim[i, j])
                if score < min_similarity:
                    break
                edge = (i, j) if i < j else (j, i)
                if edge not in edges or score > edges[edge]:
                    edges[edge] = score
                count += 1
                if count >= top_k:
                    break
        return edges

    edge_scores = build_edges(MIN_SIMILARITY, TOP_K)
    min_edges = max(5, len(base) // 2)
    if len(edge_scores) < min_edges:
        edge_scores = build_edges(FALLBACK_MIN_SIMILARITY, FALLBACK_TOP_K)

    G = nx.Graph()
    for idx, row in base.reset_index(drop=True).iterrows():
        G.add_node(idx, iso3=row["iso3"])

    for (i, j), score in edge_scores.items():
        distance = max(0.0, 1.0 - score)
        G.add_edge(i, j, weight=score, distance=distance)

    communities = []
    if G.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    if not communities:
        communities = [set(G.nodes())]

    cluster_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            cluster_map[node] = idx

    cluster_stats = (
        base.assign(cluster=base.index.map(cluster_map))
        .groupby("cluster", as_index=False)["suicide_rate"]
        .mean()
        .sort_values("suicide_rate")
    )
    ordered_clusters = cluster_stats["cluster"].tolist()
    label_map = {cluster_id: f"Cluster {chr(65 + i)}" for i, cluster_id in enumerate(ordered_clusters)}

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    pos = nx.spring_layout(G, seed=SEED, weight="weight")

    clusters = base.copy()
    clusters["cluster"] = clusters.index.map(cluster_map).fillna(-1).astype(int)
    clusters["cluster_label"] = clusters["cluster"].map(label_map)
    clusters["cluster_size"] = clusters["cluster"].map(clusters["cluster"].value_counts()).fillna(1).astype(int)
    clusters["degree_centrality"] = clusters.index.map(degree).fillna(0.0)
    clusters["betweenness_centrality"] = clusters.index.map(betweenness).fillna(0.0)
    clusters["x"] = clusters.index.map(lambda idx: float(pos.get(idx, [0.0, 0.0])[0]))
    clusters["y"] = clusters.index.map(lambda idx: float(pos.get(idx, [0.0, 0.0])[1]))

    clusters_path = REPORT_DIR / "v2_graph_clusters.csv"
    clusters.to_csv(clusters_path, index=False)

    edges = []
    for i, j, data in G.edges(data=True):
        src = clusters.iloc[i]
        tgt = clusters.iloc[j]
        edges.append(
            {
                "source_iso3": src["iso3"],
                "target_iso3": tgt["iso3"],
                "source_name": src["location_name"],
                "target_name": tgt["location_name"],
                "similarity": float(data.get("weight", 0.0)),
                "distance": float(data.get("distance", 0.0)),
                "source_x": float(src["x"]),
                "source_y": float(src["y"]),
                "target_x": float(tgt["x"]),
                "target_y": float(tgt["y"]),
            }
        )
    edges_df = pd.DataFrame(edges).sort_values("similarity", ascending=False)
    edges_path = REPORT_DIR / "v2_graph_edges.csv"
    edges_df.to_csv(edges_path, index=False)

    centrality = clusters[
        [
            "iso3",
            "location_name",
            "region_name",
            "income_group",
            "cluster_label",
            "degree_centrality",
            "betweenness_centrality",
        ]
    ].copy()
    centrality_path = REPORT_DIR / "v2_graph_centrality.csv"
    centrality.to_csv(centrality_path, index=False)

    notes = [
        "# v2 Graph Clustering Notes",
        "",
        "Cosine similarity graph built on standardized 2023 (Both) feature profiles.",
        f"- Similarity threshold: {MIN_SIMILARITY}",
        f"- Top-k neighbors per node: {TOP_K}",
        "- Community detection: greedy modularity on weighted graph.",
        "",
        "Outputs:",
        f"- {clusters_path.relative_to(REPO_ROOT)}",
        f"- {edges_path.relative_to(REPO_ROOT)}",
        f"- {centrality_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "v2_graph_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"[{VERSION}] Wrote graph clusters to {clusters_path}")


if __name__ == "__main__":
    main()
