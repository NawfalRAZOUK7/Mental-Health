#!/usr/bin/env python3
from __future__ import annotations

from project_paths import DATA_CLEAN, REPORT_DIR, REPO_ROOT, VERSION, ensure_dirs

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = numeric(
        df,
        [
            "age_standardized_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "gbd_selfharm_death_rate_male",
            "gbd_selfharm_death_rate_female",
        ],
    )
    df["gbd_selfharm_death_rate_both"] = df[
        ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
    ].mean(axis=1)

    df["region_name"] = df["region_name"].fillna("Unknown")
    df["income_group"] = df["income_group"].fillna("Unknown")
    df["data_quality"] = df["data_quality"].fillna("Unknown")

    group_cols = ["iso3", "location_name", "region_name", "income_group", "data_quality"]
    agg_cols = {
        "age_standardized_suicide_rate_2021": "mean",
        "gbd_depression_dalys_rate_both": "mean",
        "gbd_addiction_death_rate_both": "mean",
        "gbd_selfharm_death_rate_both": "mean",
    }
    df = df[group_cols + list(agg_cols.keys())].groupby(group_cols, as_index=False).agg(agg_cols)

    df = df.dropna(
        subset=[
            "age_standardized_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "gbd_selfharm_death_rate_both",
        ]
    )
    return df


def main() -> None:
    ensure_dirs()
    data_path = DATA_CLEAN / "merged_ml_country.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run 04_merge_ml.py first.")

    df = pd.read_csv(data_path)
    df = build_features(df)

    target = "age_standardized_suicide_rate_2021"
    feature_cols = [
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_selfharm_death_rate_both",
    ]
    categorical_cols = ["region_name", "income_group", "data_quality"]

    feature_table = df[
        [
            "iso3",
            "location_name",
            "region_name",
            "income_group",
            "data_quality",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "gbd_selfharm_death_rate_both",
            target,
        ]
    ].copy()
    features_path = DATA_CLEAN / "ml_baseline_features.csv"
    feature_table.to_csv(features_path, index=False)

    X = df[feature_cols + categorical_cols]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(
            [
                ("prep", preprocessor),
                ("model", RandomForestRegressor(n_estimators=400, random_state=42)),
            ]
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        mae_scores = -scores["test_mae"]
        r2_scores = scores["test_r2"]
        cv_results.append(
            {
                "model": name,
                "mae_mean": mae_scores.mean(),
                "mae_std": mae_scores.std(),
                "r2_mean": r2_scores.mean(),
                "r2_std": r2_scores.std(),
            }
        )

    cv_df = pd.DataFrame(cv_results).sort_values("mae_mean")
    cv_path = REPORT_DIR / "ml_baseline_cv.csv"
    cv_df.to_csv(cv_path, index=False)

    results = []
    predictions = df[["iso3", "location_name"]].copy()
    predictions["actual"] = y

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred),
            }
        )

        full_pred = model.predict(X)
        predictions[f"{name.lower()}_pred"] = full_pred

    results_df = pd.DataFrame(results).sort_values("mae")
    results_path = REPORT_DIR / "ml_baseline_results.csv"
    results_df.to_csv(results_path, index=False)

    preds_path = DATA_CLEAN / "ml_baseline_predictions.csv"
    predictions.to_csv(preds_path, index=False)

    rf_model = models["RandomForest"].named_steps["model"]
    feature_names = models["RandomForest"].named_steps["prep"].get_feature_names_out()
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_path = REPORT_DIR / "ml_feature_importance.csv"
    importances.to_csv(importance_path, index=False)

    results_table = ["| model | mae | r2 |", "| --- | --- | --- |"]
    for _, row in results_df.iterrows():
        results_table.append(f"| {row['model']} | {row['mae']:.4f} | {row['r2']:.4f} |")

    cv_table = ["| model | mae_mean | mae_std | r2_mean | r2_std |", "| --- | --- | --- | --- | --- |"]
    for _, row in cv_df.iterrows():
        cv_table.append(
            f"| {row['model']} | {row['mae_mean']:.4f} | {row['mae_std']:.4f} | {row['r2_mean']:.4f} | {row['r2_std']:.4f} |"
        )

    report_lines = [
        "# ML Baseline Report",
        "",
        "## Target",
        f"- {target}",
        "",
        "## Features",
        "- gbd_depression_dalys_rate_both (DALYs rate)",
        "- gbd_addiction_death_rate_both (Deaths rate)",
        "- gbd_selfharm_death_rate_both (Deaths rate; mean of male/female)",
        "- region_name, income_group, data_quality (one-hot)",
        "",
        "## Data preparation",
        "- Aggregated to one row per country (mean across age groups) to align with age-standardized target.",
        f"- Rows used: {len(df)} countries",
        "",
        "## Holdout Results (25% test)",
        *results_table,
        "",
        "## Cross-validation (5-fold)",
        *cv_table,
        "",
        "## Outputs",
        f"- {results_path.relative_to(REPO_ROOT)}",
        f"- {cv_path.relative_to(REPO_ROOT)}",
        f"- {preds_path.relative_to(REPO_ROOT)}",
        f"- {features_path.relative_to(REPO_ROOT)}",
        f"- {importance_path.relative_to(REPO_ROOT)}",
    ]
    report_path = REPORT_DIR / "ml_baseline.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[{VERSION}] Wrote results to:")
    print(f"- {results_path}")
    print(f"- {cv_path}")
    print(f"- {preds_path}")
    print(f"- {features_path}")
    print(f"- {importance_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
