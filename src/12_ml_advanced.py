#!/usr/bin/env python3
"""Advanced ML on real enriched data: tuned gradient boosting + SHAP + conformal intervals.

Upgrades over 06/11:
  1. LightGBM with Optuna hyperparameter search inside a NESTED cross-validation
     (honest generalization estimate, no tuning leakage).
  2. SHAP explanations (modern per-feature + per-country attribution).
  3. Split-conformal prediction intervals with empirical coverage.

Outputs (v1/report/): ml_advanced_metrics.csv, ml_advanced_shap.csv,
ml_advanced_conformal.csv, ml_advanced.md
Figures (report_latex/figures/): fig_v1_shap_summary.png
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import optuna
    import shap
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
except ImportError as exc:
    raise SystemExit("Run: pip install -r requirements-advanced.txt") from exc

import theme
from advanced_common import TARGET, load_enriched
from project_paths import REPO_ROOT, REPORT_DIR, VERSION, ensure_dirs

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
RNG = 42


def encode(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    X = df[num].copy()
    for c in cat:
        X[c] = df[c].astype("category")
    return X


def tune(X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RNG,
            "verbose": -1,
        }
        inner = KFold(3, shuffle=True, random_state=RNG)
        maes = []
        for tr, va in inner.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[va], model.predict(X.iloc[va])))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def main() -> None:
    ensure_dirs()
    theme.apply_matplotlib()
    df, num, cat = load_enriched()
    X, y = encode(df, num, cat), df[TARGET]
    print(f"[{VERSION}] countries={len(df)} numeric_features={len(num)}")

    # ---- Nested CV: outer folds estimate generalization; inner folds tune. ----
    outer = KFold(5, shuffle=True, random_state=RNG)
    outer_mae, outer_r2 = [], []
    for tr, te in outer.split(X):
        best = tune(X.iloc[tr], y.iloc[tr], n_trials=30)
        model = lgb.LGBMRegressor(**best, random_state=RNG, verbose=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        outer_mae.append(mean_absolute_error(y.iloc[te], pred))
        outer_r2.append(r2_score(y.iloc[te], pred))

    metrics = pd.DataFrame([
        {"model": "LightGBM (nested-CV, tuned)", "mae_mean": np.mean(outer_mae),
         "mae_std": np.std(outer_mae), "r2_mean": np.mean(outer_r2), "r2_std": np.std(outer_r2)},
    ])
    metrics.to_csv(REPORT_DIR / "ml_advanced_metrics.csv", index=False)

    # ---- Final model on all data (for SHAP + conformal), tuned once. ----
    best = tune(X, y, n_trials=40)
    final = lgb.LGBMRegressor(**best, random_state=RNG, verbose=-1)
    final.fit(X, y)

    # ---- SHAP ----
    explainer = shap.TreeExplainer(final)
    sv = explainer.shap_values(X)
    shap_imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
                .sort_values("mean_abs_shap", ascending=False))
    shap_imp.to_csv(REPORT_DIR / "ml_advanced_shap.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(9, 5))
    fig_dir = REPO_ROOT / "report_latex" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_v1_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Split-conformal prediction intervals (90%) ----
    idx = np.arange(len(X))
    rng = np.random.default_rng(RNG)
    rng.shuffle(idx)
    cut = int(0.75 * len(idx))
    tr_idx, cal_idx = idx[:cut], idx[cut:]
    cmodel = lgb.LGBMRegressor(**best, random_state=RNG, verbose=-1)
    cmodel.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    cal_resid = np.abs(y.iloc[cal_idx].to_numpy() - cmodel.predict(X.iloc[cal_idx]))
    q = float(np.quantile(cal_resid, 0.90))
    full_pred = final.predict(X)
    lower, upper = full_pred - q, full_pred + q
    coverage = float(np.mean((y.to_numpy() >= lower) & (y.to_numpy() <= upper)))
    conf = df[["iso3", "location_name", TARGET]].copy()
    conf["pred"] = full_pred
    conf["lower_90"] = lower
    conf["upper_90"] = upper
    conf.to_csv(REPORT_DIR / "ml_advanced_conformal.csv", index=False)

    top = shap_imp.head(5)["feature"].tolist()
    lines = [
        "# Advanced ML (real data): tuned LightGBM + SHAP + conformal",
        "",
        f"Countries: {len(df)}. Target: WHO age-standardized suicide rate.",
        "",
        "## Nested cross-validation (honest, tuning inside folds)",
        f"- LightGBM MAE = {np.mean(outer_mae):.2f} (+/- {np.std(outer_mae):.2f})",
        f"- LightGBM R2  = {np.mean(outer_r2):.3f} (+/- {np.std(outer_r2):.3f})",
        "",
        "## SHAP — top drivers",
        f"- {', '.join(top)}",
        "",
        "## Conformal prediction intervals (target 90%)",
        f"- Interval half-width: +/- {q:.2f} per 100k",
        f"- Empirical coverage: {coverage:.1%}",
        "",
        "## Outputs",
        "- ml_advanced_metrics.csv, ml_advanced_shap.csv, ml_advanced_conformal.csv",
        "- report_latex/figures/fig_v1_shap_summary.png",
    ]
    (REPORT_DIR / "ml_advanced.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{VERSION}] nested-CV R2={np.mean(outer_r2):.3f} | conformal coverage={coverage:.1%}")


if __name__ == "__main__":
    main()
