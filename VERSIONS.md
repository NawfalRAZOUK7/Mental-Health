# Project Versions

This repo supports multiple output versions while sharing `data_raw/` and `src/app.py`.

## Versions
- v0: simple static visuals (no web app)
- v1: validated pipeline (current)
- v2: synthetic/advanced demo (generated data)
- v3: interactive risk estimator (v1/v2 selectable)

## Layout
- `v0/`, `v1/`, `v2/` are version roots.
- `v*/data_clean/` and `v*/report/` store versioned outputs.
- `data_raw/` stays shared across all versions.

## How to run
| Version | Build outputs | Run dashboard |
| --- | --- | --- |
| v0 | `MHP_VERSION=v0 python src/v0_visuals.py` | `python scripts/run_app.py --version v0` |
| v1 | `python scripts/run_v1_pipeline.py` | `python scripts/run_app.py --version v1` |
| v2 | `python scripts/run_v2_pipeline.py` | `python scripts/run_app.py --version v2` |
| v3 | `python scripts/run_v3_pipeline.py` | `python scripts/run_app.py --version v3` |

## Advanced analytics (real data, v1)
Run the modern ML / data-mining suite on the covariate-enriched real data:
```bash
pip install -r requirements-advanced.txt
python scripts/run_advanced.py
```
This produces: tuned LightGBM (Optuna, nested CV) + SHAP + conformal intervals,
a hierarchical mixed-effects model (ICC), a UMAP embedding, a country-similarity
network, and subgroup / association-rule mining.

## Optional add-ons (opt-in, isolated dependencies)
| Add-on | Enable | Extra install |
| --- | --- | --- |
| Spatial autocorrelation (Moran's I / LISA) | `MHV_SPATIAL=1 python scripts/run_advanced.py` | `libpysal esda spreg` |
| Deep global forecasting (N-BEATS + darts TFT) | `MHV_DEEP=1 [MHV_DEEP_BACKEND=torch\|darts\|all] python src/v2_global_forecast.py` | `pip install -r requirements-deep.txt` |

The base install, dashboards, and deployment never require these; each skips
gracefully when its dependencies are absent.

## Serving & product
- **Prediction API** — FastAPI service with conformal intervals (`api/`, `docker compose up`).
- **Website** — Next.js/React static site presenting the project with a live predictor,
  version links, and a rule-based guide chatbot (`web/`).
- **Bilingual** — the website, chatbot, and all Streamlit UI text (v0–v3) support EN/FR.

## Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-v2.txt        # v2 advanced features
pip install -r requirements-advanced.txt  # v1 real-data ML / data-mining suite
# optional: requirements-deep.txt (deep forecasting), libpysal/esda/spreg (spatial)
```
