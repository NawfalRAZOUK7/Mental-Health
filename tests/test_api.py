"""Smoke tests for the prediction API.

Skips automatically if FastAPI/httpx are not installed (e.g. in the lightweight
CI job that only installs core requirements), so it never breaks the pipeline.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

os.environ.setdefault("MHP_VERSION", "v1")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def test_endpoints() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json()["model_ready"] is True

        countries = client.get("/countries").json()
        assert countries["count"] > 100

        fra = client.get("/predict/FRA").json()
        assert fra["iso3"] == "FRA"
        assert 0 < fra["predicted_suicide_rate"] < 60
        assert fra["lower_90"] <= fra["predicted_suicide_rate"] <= fra["upper_90"]

        assert client.get("/predict/ZZZ").status_code == 404

        post = client.post("/predict", json={"region_name": "European Region",
                                             "income_group": "HI",
                                             "alcohol_litres_per_capita": 12.0})
        assert "predicted_suicide_rate" in post.json()
