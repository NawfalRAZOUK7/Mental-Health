"""Prediction model for the API.

Leakage-free predictor of the WHO age-standardized suicide rate from independent
drivers (depression & addiction burden + World Bank socioeconomic covariates +
geography). Uses a pure scikit-learn HistGradientBoostingRegressor (no torch /
lightgbm, so the container stays light and dependency-safe) and split-conformal
90% intervals. Self-harm is deliberately NOT a feature (see 09_source_agreement.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MHP_VERSION", "v1")

from advanced_common import TARGET, load_enriched  # noqa: E402

RNG = 42


class Predictor:
    """Trains once on startup; serves per-country and per-feature predictions."""

    def __init__(self) -> None:
        df, num, cat = load_enriched()
        self.df = df
        self.num_features = num
        self.cat_features = cat
        self.feature_order = num + cat

        X, y = df[self.feature_order], df[TARGET]
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        pre = ColumnTransformer([
            ("num", num_pipe, num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        self.model = Pipeline([("prep", pre),
                               ("model", HistGradientBoostingRegressor(random_state=RNG))])

        # Split-conformal 90% interval half-width.
        idx = np.arange(len(df))
        np.random.default_rng(RNG).shuffle(idx)
        cut = int(0.75 * len(idx))
        tr, cal = idx[:cut], idx[cut:]
        self.model.fit(X.iloc[tr], y.iloc[tr])
        resid = np.abs(y.iloc[cal].to_numpy() - self.model.predict(X.iloc[cal]))
        self.q90 = float(np.quantile(resid, 0.90))

        # Refit on all data for the served model.
        self.model.fit(X, y)

        self._by_iso = {r["iso3"]: r for r in df.to_dict("records")}
        self.countries = (df[["iso3", "location_name"]]
                          .drop_duplicates().sort_values("location_name")
                          .to_dict("records"))

    def _interval(self, pred: float) -> dict:
        return {
            "predicted_suicide_rate": round(pred, 2),
            "lower_90": round(pred - self.q90, 2),
            "upper_90": round(pred + self.q90, 2),
            "interval_half_width": round(self.q90, 2),
        }

    def predict_country(self, iso3: str) -> dict | None:
        row = self._by_iso.get(iso3.upper())
        if row is None:
            return None
        xr = pd.DataFrame([row])[self.feature_order]
        pred = float(self.model.predict(xr)[0])
        out = {"iso3": iso3.upper(), "name": row["location_name"],
               "actual_suicide_rate": round(float(row[TARGET]), 2)}
        out.update(self._interval(pred))
        return out

    def predict_features(self, feats: dict) -> dict:
        xr = pd.DataFrame([feats])
        for col in self.feature_order:
            if col not in xr.columns:
                xr[col] = np.nan
        for col in self.cat_features:
            xr[col] = xr[col].fillna("Unknown")
        pred = float(self.model.predict(xr[self.feature_order])[0])
        return self._interval(pred)
