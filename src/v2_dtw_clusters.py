#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_dtw
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
except ImportError as exc:
    raise SystemExit(
        "tslearn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    silhouette_score = None

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs


K_OPTIONS = [3, 4, 5, 6]
MAX_SILHOUETTE_SAMPLES = 250


def prepare_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    df = df[(df["sex_name"] == "Both") & df["year"].notna()].copy()
    df["suicide_rate"] = pd.to_numeric(df["suicide_rate"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]

    years = sorted(df["year"].unique())
    series = df.pivot_table(
        index="iso3",
        columns="year",
        values="suicide_rate",
        aggfunc="mean",
    ).reindex(columns=years)
    series = series.interpolate(axis=1, limit_direction="both")
    series = series.dropna()
    if series.empty:
        raise SystemExit("No complete time series available for DTW clustering.")

    meta = (
        df.groupby("iso3")[["location_name", "region_name", "income_group"]]
        .first()
        .loc[series.index]
        .reset_index()
    )
    return series, meta, years


def main() -> None:
    ensure_dirs()
    if VERSION != "v2":
        print(f"Warning: MHP_VERSION is {VERSION}; outputs go to {REPORT_DIR}")

    path = DATA_CLEAN / "synth_country_year.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(path)
    series, meta, years = prepare_series(df)

    X = series.to_numpy()
    scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
    X_scaled = scaler.fit_transform(X)

    dist_matrix = None
    if silhouette_score is not None and len(series) <= MAX_SILHOUETTE_SAMPLES:
        dist_matrix = cdist_dtw(X_scaled)

    k_metrics = []
    for k in K_OPTIONS:
        if len(series) <= k:
            continue
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            random_state=42,
            n_init=2,
            max_iter=20,
        )
        labels = model.fit_predict(X_scaled)
        silhouette = float("nan")
        if dist_matrix is not None and len(set(labels)) > 1:
            silhouette = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
        k_metrics.append(
            {"k": k, "silhouette": silhouette, "inertia": float(model.inertia_)}
        )

    metrics_df = pd.DataFrame(k_metrics)
    if metrics_df.empty:
        raise SystemExit("Not enough rows for DTW clustering.")

    if metrics_df["silhouette"].notna().any():
        best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    else:
        best_k = int(metrics_df.loc[metrics_df["inertia"].idxmin(), "k"])

    final_model = TimeSeriesKMeans(
        n_clusters=best_k,
        metric="dtw",
        random_state=42,
        n_init=2,
        max_iter=30,
    )
    labels = final_model.fit_predict(X_scaled)

    cluster_df = meta.copy()
    cluster_df["cluster"] = labels
    cluster_df["mean_rate"] = series.mean(axis=1).to_numpy()
    order = cluster_df.groupby("cluster")["mean_rate"].mean().sort_values().index.tolist()
    label_map = {cluster_id: f"Cluster {chr(65 + idx)}" for idx, cluster_id in enumerate(order)}
    cluster_df["cluster_label"] = cluster_df["cluster"].map(label_map)

    centers = []
    for cluster_id in order:
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        center_series = np.nanmean(X[mask], axis=0)
        row = {"cluster": int(cluster_id), "cluster_label": label_map[cluster_id]}
        row.update({str(year): float(val) for year, val in zip(years, center_series)})
        centers.append(row)
    centers_df = pd.DataFrame(centers)

    clusters_path = REPORT_DIR / "v2_dtw_clusters.csv"
    cluster_df.to_csv(clusters_path, index=False)

    centers_path = REPORT_DIR / "v2_dtw_cluster_centers.csv"
    centers_df.to_csv(centers_path, index=False)

    metrics_path = REPORT_DIR / "v2_dtw_k_selection.csv"
    metrics_df.to_csv(metrics_path, index=False)

    notes = [
        "# v2 DTW Clustering Notes",
        "",
        "DTW clustering on 2000-2023 suicide_rate time series (Both sexes).",
        f"- Selected k={best_k}.",
        "",
        "Preprocessing:",
        "- Per-country interpolation over years to fill missing values.",
        "- Per-series z-score standardization before DTW.",
        "",
        "Outputs:",
        f"- {clusters_path.relative_to(REPO_ROOT)}",
        f"- {centers_path.relative_to(REPO_ROOT)}",
        f"- {metrics_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "v2_dtw_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"[{VERSION}] Wrote DTW clusters to {clusters_path}")


if __name__ == "__main__":
    main()
