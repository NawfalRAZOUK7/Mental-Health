#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


FEATURES = [
    "slope",
    "volatility",
    "peak_value",
    "last5_change",
    "mean_rate",
]


def compute_trajectory_features(group: pd.DataFrame) -> dict[str, float]:
    series = group.dropna(subset=["suicide_rate"]).copy()
    series = series.sort_values("year")
    years = series["year"].to_numpy()
    values = series["suicide_rate"].to_numpy()
    if len(values) < 5:
        return {}

    slope = np.polyfit(years, values, 1)[0]
    volatility = float(np.std(values, ddof=0))
    peak_idx = int(np.argmax(values))
    peak_year = int(years[peak_idx])
    peak_value = float(values[peak_idx])
    mean_rate = float(np.mean(values))
    recent = values[-5:] if len(values) >= 5 else values
    last5_change = float(np.mean(recent) - mean_rate)

    return {
        "slope": float(slope),
        "volatility": volatility,
        "peak_year": peak_year,
        "peak_value": peak_value,
        "last5_change": last5_change,
        "mean_rate": mean_rate,
    }


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {DATA_CLEAN}")

    path = DATA_CLEAN / "synth_country_year.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["suicide_rate"] = pd.to_numeric(df["suicide_rate"], errors="coerce")
    df = df[(df["sex_name"] == "Both") & df["year"].notna()]

    rows = []
    group_cols = ["iso3", "location_name", "region_name", "income_group"]
    for keys, group in df.groupby(group_cols, sort=False):
        features = compute_trajectory_features(group)
        if not features:
            continue
        row = dict(zip(group_cols, keys))
        row.update(features)
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        raise SystemExit("No trajectory features generated.")

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df[FEATURES])

    k_metrics = []
    for k in [3, 4, 5, 6]:
        if len(feat_df) <= k:
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = float("nan")
        if len(set(labels)) > 1:
            score = float(silhouette_score(X, labels))
        k_metrics.append({"k": k, "silhouette": score, "inertia": float(model.inertia_)})

    metrics_df = pd.DataFrame(k_metrics)
    if metrics_df.empty:
        raise SystemExit("Not enough rows for clustering.")

    best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    feat_df["cluster"] = labels

    order = (
        feat_df.groupby("cluster")["mean_rate"].mean().sort_values().index.tolist()
    )
    label_map = {cluster_id: f"Cluster {chr(65 + idx)}" for idx, cluster_id in enumerate(order)}
    feat_df["cluster_label"] = feat_df["cluster"].map(label_map)

    centers_scaled = pd.DataFrame(model.cluster_centers_, columns=[f"{c}_z" for c in FEATURES])
    centers_original = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    centers = pd.concat([centers_original, centers_scaled], axis=1)
    centers["cluster"] = centers.index
    centers["cluster_label"] = centers["cluster"].map(label_map)
    centers["count"] = centers["cluster"].map(feat_df["cluster"].value_counts().to_dict()).fillna(0).astype(int)

    clusters_path = DATA_CLEAN / "v2_trajectory_clusters.csv"
    feat_df.to_csv(clusters_path, index=False)

    centers_path = REPORT_DIR / "v2_trajectory_cluster_centers.csv"
    centers.to_csv(centers_path, index=False)

    metrics_path = REPORT_DIR / "v2_trajectory_k_selection.csv"
    metrics_df.to_csv(metrics_path, index=False)

    notes = [
        "# v2 Trajectory Clustering Notes",
        "",
        "Clustering uses trajectory features from 2000-2023 suicide_rate time series.",
        f"- Selected k={best_k} by silhouette score.",
        "",
        "Features:",
        "- slope (linear trend)",
        "- volatility (std dev)",
        "- peak_value and peak_year",
        "- last5_change (last 5y mean - long-run mean)",
        "- mean_rate (overall mean)",
        "",
        "Outputs:",
        f"- {clusters_path.relative_to(REPO_ROOT)}",
        f"- {centers_path.relative_to(REPO_ROOT)}",
        f"- {metrics_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "v2_trajectory_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"[{VERSION}] Wrote trajectory clusters to {clusters_path}")
    print(f"[{VERSION}] Wrote centers to {centers_path}")


if __name__ == "__main__":
    main()
