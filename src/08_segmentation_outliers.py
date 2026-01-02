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
    "age_standardized_suicide_rate_2021",
    "gbd_depression_dalys_rate_both",
    "gbd_addiction_death_rate_both",
    "gbd_selfharm_death_rate_both",
]

FEATURE_LABELS = {
    "age_standardized_suicide_rate_2021": "suicide_rate",
    "gbd_depression_dalys_rate_both": "depression_dalys_rate",
    "gbd_addiction_death_rate_both": "addiction_death_rate",
    "gbd_selfharm_death_rate_both": "selfharm_death_rate",
}


def select_best_k(X: np.ndarray, k_values: list[int]) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        silhouette = float("nan")
        try:
            silhouette = float(silhouette_score(X, labels))
        except ValueError:
            silhouette = float("nan")
        rows.append({"k": k, "inertia": inertia, "silhouette": silhouette})

    metrics = pd.DataFrame(rows)
    if metrics["silhouette"].notna().any():
        best_row = metrics.loc[metrics["silhouette"].idxmax()]
        best_k = int(best_row["k"])
    else:
        best_k = k_values[0]
    return best_k, metrics


def build_outlier_reason(row: pd.Series, z_cols: list[str], threshold: float) -> str:
    reasons = []
    for feature, z_col in zip(FEATURES, z_cols):
        z_val = row.get(z_col)
        if pd.isna(z_val):
            continue
        if abs(float(z_val)) >= threshold:
            direction = "high" if float(z_val) > 0 else "low"
            reasons.append(f"{FEATURE_LABELS[feature]} ({direction})")
    return "; ".join(reasons)


def main() -> None:
    ensure_dirs()
    feature_path = DATA_CLEAN / "ml_baseline_features.csv"
    if not feature_path.exists():
        raise SystemExit(f"Missing {feature_path}. Run 06_ml_baseline.py first.")

    df = pd.read_csv(feature_path)
    df = df.dropna(subset=FEATURES).copy()
    if len(df) < 10:
        raise SystemExit("Not enough rows for clustering. Check feature completeness.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    best_k, k_metrics = select_best_k(X, [3, 4, 5, 6])
    k_metrics_path = REPORT_DIR / "segmentation_k_selection.csv"
    k_metrics.to_csv(k_metrics_path, index=False)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    df["cluster"] = labels

    cluster_summary = df.groupby("cluster")[FEATURES].mean().reset_index()
    ordered = cluster_summary.sort_values(
        "age_standardized_suicide_rate_2021"
    )["cluster"].tolist()
    label_map = {cluster_id: f"Cluster {chr(65 + idx)}" for idx, cluster_id in enumerate(ordered)}
    df["cluster_label"] = df["cluster"].map(label_map)

    scaled_cols = [f"{feature}_z" for feature in FEATURES]
    scaled_df = pd.DataFrame(X, columns=scaled_cols)
    df_scaled = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    centers_scaled = pd.DataFrame(
        model.cluster_centers_, columns=scaled_cols
    )
    centers_original = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEATURES,
    )
    centers = pd.concat([centers_original, centers_scaled], axis=1)
    centers["cluster"] = centers.index
    centers["cluster_label"] = centers["cluster"].map(label_map)
    counts = df["cluster"].value_counts().to_dict()
    centers["count"] = centers["cluster"].map(counts).fillna(0).astype(int)

    centers_path = REPORT_DIR / "segmentation_cluster_centers.csv"
    centers.to_csv(centers_path, index=False)

    threshold = 2.5
    df_scaled["outlier_score"] = df_scaled[scaled_cols].abs().max(axis=1)
    df_scaled["is_outlier"] = df_scaled["outlier_score"] >= threshold
    df_scaled["outlier_reason"] = df_scaled.apply(
        build_outlier_reason, axis=1, args=(scaled_cols, threshold)
    )

    df_scaled_path = DATA_CLEAN / "segmentation_features.csv"
    df_scaled.to_csv(df_scaled_path, index=False)

    clusters_path = DATA_CLEAN / "segmentation_clusters.csv"
    df_scaled.to_csv(clusters_path, index=False)

    outliers = df_scaled[df_scaled["is_outlier"]].copy()
    outliers = outliers.sort_values("outlier_score", ascending=False)

    outlier_cols = [
        "iso3",
        "location_name",
        "region_name",
        "income_group",
        "outlier_score",
        "outlier_reason",
    ] + FEATURES
    outliers_path = REPORT_DIR / "outliers_table.csv"
    outliers[outlier_cols].to_csv(outliers_path, index=False)

    notes_lines = [
        "# Segmentation + Outliers Notes",
        "",
        "## Features",
        "- age_standardized_suicide_rate_2021 (WHO)",
        "- gbd_depression_dalys_rate_both (GBD 2023)",
        "- gbd_addiction_death_rate_both (GBD 2023, substance use disorders)",
        "- gbd_selfharm_death_rate_both (GBD 2023, mean of male/female)",
        "",
        "## Clustering",
        f"- KMeans with k selected from {k_metrics['k'].tolist()} by max silhouette.",
        f"- Selected k = {best_k}.",
        "",
        "## Outliers",
        f"- Z-score threshold: {threshold}",
        "- Outlier reason lists which features are high/low vs population.",
        "",
        "## Outputs",
        f"- {df_scaled_path.relative_to(REPO_ROOT)}",
        f"- {clusters_path.relative_to(REPO_ROOT)}",
        f"- {centers_path.relative_to(REPO_ROOT)}",
        f"- {k_metrics_path.relative_to(REPO_ROOT)}",
        f"- {outliers_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "segmentation_outliers.md"
    notes_path.write_text("\n".join(notes_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote segmentation outputs:")
    print(f"- {df_scaled_path}")
    print(f"- {clusters_path}")
    print(f"- {centers_path}")
    print(f"- {k_metrics_path}")
    print(f"- {outliers_path}")
    print(f"- {notes_path}")


if __name__ == "__main__":
    main()
