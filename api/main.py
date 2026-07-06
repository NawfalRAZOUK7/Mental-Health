"""FastAPI service exposing the leakage-free suicide-rate predictor.

Run locally:
    pip install -r api/requirements.txt
    uvicorn api.main:app --reload
Then open http://localhost:8000/docs for the interactive Swagger UI.

Educational / non-clinical. Predictions are model estimates, not medical advice.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.model import Predictor

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor()  # train once at startup
    yield


app = FastAPI(
    title="Mental Health Viz — Prediction API",
    description="Leakage-free estimate of the national age-standardized suicide rate "
                "with 90% conformal intervals. Educational use only — not clinical.",
    version="1.0.0",
    lifespan=lifespan,
)


class FeatureInput(BaseModel):
    region_name: str = Field("Unknown", examples=["European Region"])
    income_group: str = Field("Unknown", examples=["HI"])
    depression_dalys_rate: float | None = Field(None, examples=[600.0])
    addiction_death_rate: float | None = Field(None, examples=[3.0])
    gdp_per_capita_usd: float | None = Field(None, examples=[30000.0])
    unemployment_pct: float | None = Field(None, examples=[6.0])
    health_exp_per_capita_usd: float | None = Field(None, examples=[3000.0])
    alcohol_litres_per_capita: float | None = Field(None, examples=[10.0])
    gini_index: float | None = Field(None, examples=[33.0])
    urban_population_pct: float | None = Field(None, examples=[75.0])
    life_expectancy_years: float | None = Field(None, examples=[80.0])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": predictor is not None}


@app.get("/countries")
def countries() -> dict:
    if predictor is None:
        raise HTTPException(503, "Model not ready")
    return {"count": len(predictor.countries), "countries": predictor.countries}


@app.get("/predict/{iso3}")
def predict_country(iso3: str) -> dict:
    if predictor is None:
        raise HTTPException(503, "Model not ready")
    result = predictor.predict_country(iso3)
    if result is None:
        raise HTTPException(404, f"Unknown country code: {iso3}")
    return {"disclaimer": "Educational estimate, not medical advice.", **result}


@app.post("/predict")
def predict_features(payload: FeatureInput) -> dict:
    if predictor is None:
        raise HTTPException(503, "Model not ready")
    result = predictor.predict_features(payload.model_dump())
    return {"disclaimer": "Educational estimate, not medical advice.", **result}
