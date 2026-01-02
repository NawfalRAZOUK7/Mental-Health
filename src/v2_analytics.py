#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
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
MODEL_FEATURE_COLS = [
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]


def select_best_k(X: np.ndarray, k_values: list[int]) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        silhouette = float("nan")
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(X, labels))
        rows.append({"k": k, "inertia": inertia, "silhouette": silhouette})
    metrics = pd.DataFrame(rows)
    if metrics["silhouette"].notna().any():
        best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    else:
        best_k = k_values[0]
    return best_k, metrics


def build_outlier_reason(row: pd.Series, z_cols: list[str]) -> str:
    reasons = []
    for feature, z_col in zip(FEATURE_COLS, z_cols):
        z_val = row.get(z_col)
        if pd.isna(z_val):
            continue
        if abs(float(z_val)) >= 2.0:
            direction = "high" if float(z_val) > 0 else "low"
            reasons.append(f"{feature} ({direction})")
    return "; ".join(reasons)


def main() -> None:
    ensure_dirs()
    synth_path = DATA_CLEAN / "synth_country_year.csv"
    if not synth_path.exists():
        raise SystemExit(f"Missing {synth_path}. Run src/v2_generate_synth.py first.")

    df = pd.read_csv(synth_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in FEATURE_COLS + ["population", "risk_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    base = df[(df["year"] == 2023) & (df["sex_name"] == "Both")].copy()
    base = base.dropna(subset=FEATURE_COLS)
    if base.empty:
        raise SystemExit("No data available for clustering/outliers.")

    scaler = StandardScaler()
    X = scaler.fit_transform(base[FEATURE_COLS])

    best_k, k_metrics = select_best_k(X, [3, 4, 5, 6])
    k_path = REPORT_DIR / "v2_k_selection.csv"
    k_metrics.to_csv(k_path, index=False)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    base["cluster"] = labels

    cluster_order = base.groupby("cluster")["suicide_rate"].mean().sort_values().index.tolist()
    label_map = {cluster_id: f"Cluster {chr(65 + idx)}" for idx, cluster_id in enumerate(cluster_order)}
    base["cluster_label"] = base["cluster"].map(label_map)

    centers_scaled = pd.DataFrame(model.cluster_centers_, columns=[f"{c}_z" for c in FEATURE_COLS])
    centers_original = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLS)
    centers = pd.concat([centers_original, centers_scaled], axis=1)
    centers["cluster"] = centers.index
    centers["cluster_label"] = centers["cluster"].map(label_map)
    centers["count"] = centers["cluster"].map(base["cluster"].value_counts().to_dict()).fillna(0).astype(int)
    centers_path = REPORT_DIR / "v2_cluster_centers.csv"
    centers.to_csv(centers_path, index=False)

    clusters_path = DATA_CLEAN / "v2_clusters.csv"
    base.to_csv(clusters_path, index=False)

    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)
    scores = -iso_model.decision_function(X)
    base["outlier_score"] = scores
    base["is_outlier"] = iso_model.predict(X) == -1

    z_cols = [f"{c}_z" for c in FEATURE_COLS]
    z_df = pd.DataFrame(X, columns=z_cols)
    base = pd.concat([base.reset_index(drop=True), z_df], axis=1)
    base["outlier_reason"] = base.apply(build_outlier_reason, axis=1, args=(z_cols,))

    outliers_path = REPORT_DIR / "v2_outliers.csv"
    base.sort_values("outlier_score", ascending=False).to_csv(outliers_path, index=False)

    region_path = DATA_CLEAN / "synth_region_year.csv"
    region_df = pd.read_csv(region_path)
    region_df["year"] = pd.to_numeric(region_df["year"], errors="coerce")
    region_df["suicide_rate"] = pd.to_numeric(region_df["suicide_rate"], errors="coerce")
    region_df = region_df[(region_df["sex_name"] == "Both") & region_df["year"].notna()]
    region_df = region_df.dropna(subset=["suicide_rate"])

    forecast_rows = []
    for region_name, group in region_df.groupby("region_name"):
        group = group.sort_values("year")
        years = group["year"].to_numpy()
        vals = group["suicide_rate"].to_numpy()
        if len(years) < 5:
            continue
        reg = LinearRegression()
        reg.fit(years.reshape(-1, 1), vals)
        for year in range(2024, 2031):
            pred = float(reg.predict(np.array([[year]]))[0])
            forecast_rows.append(
                {
                    "region_name": region_name,
                    "year": year,
                    "suicide_rate": max(0.0, pred),
                    "type": "forecast",
                }
            )
        actual = group[["region_name", "year", "suicide_rate"]].copy()
        actual["type"] = "actual"
        forecast_rows.extend(actual.to_dict("records"))

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path = REPORT_DIR / "v2_forecast_region.csv"
    forecast_df.to_csv(forecast_path, index=False)

    train = df[(df["sex_name"] == "Both") & (df["year"] >= 2010)].copy()
    train = train.dropna(subset=MODEL_FEATURE_COLS + ["suicide_rate"])
    X_train = train[MODEL_FEATURE_COLS].to_numpy()
    y_train = train["suicide_rate"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_scaled, y_train)
    preds = lr.predict(X_scaled)
    metrics = {
        "r2": r2_score(y_train, preds),
        "mae": mean_absolute_error(y_train, preds),
        "rows": len(train),
    }
    metrics_path = REPORT_DIR / "v2_model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    coeff_rows = [
        {
            "feature": "_intercept",
            "coef": lr.intercept_,
            "mean": "",
            "scale": "",
        }
    ]
    for feature, coef, mean, scale in zip(MODEL_FEATURE_COLS, lr.coef_, scaler.mean_, scaler.scale_):
        coeff_rows.append(
            {
                "feature": feature,
                "coef": coef,
                "mean": mean,
                "scale": scale,
            }
        )
    coeff_path = REPORT_DIR / "v2_model_coeffs.csv"
    pd.DataFrame(coeff_rows).to_csv(coeff_path, index=False)

    notes = [
        "# v2 Analytics Notes",
        "",
        f"- Clustering: KMeans with k={best_k} selected by silhouette.",
        "- Outliers: IsolationForest (contamination=0.05).",
        "- Forecast: linear regression by region (2024-2030).",
        "- Scenario model: linear regression on standardized features (2010+).",
        "",
        "Outputs:",
        f"- {clusters_path.relative_to(REPO_ROOT)}",
        f"- {centers_path.relative_to(REPO_ROOT)}",
        f"- {k_path.relative_to(REPO_ROOT)}",
        f"- {outliers_path.relative_to(REPO_ROOT)}",
        f"- {forecast_path.relative_to(REPO_ROOT)}",
        f"- {coeff_path.relative_to(REPO_ROOT)}",
        f"- {metrics_path.relative_to(REPO_ROOT)}",
    ]
    notes_path = REPORT_DIR / "v2_analytics_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"[{VERSION}] Wrote clusters to {clusters_path}")
    print(f"[{VERSION}] Wrote centers to {centers_path}")
    print(f"[{VERSION}] Wrote outliers to {outliers_path}")
    print(f"[{VERSION}] Wrote forecast to {forecast_path}")
    print(f"[{VERSION}] Wrote model coeffs to {coeff_path}")


if __name__ == "__main__":
    main()
