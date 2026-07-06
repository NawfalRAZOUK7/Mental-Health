# `src/` — source code map

Modules are grouped by purpose. The **numbered files form the v1 real-data
pipeline** and run in order (each step's output feeds the next); the `v2_*` /
`v3_*` files are that version's analytics; the rest are shared or app code.

## v1 pipeline (real data) — run in order

| Step | File | Purpose |
| --- | --- | --- |
| 00 | `00_inventory.py` | Inventory raw sources |
| 01 | `01_country_mapping.py` | ISO3 country mapping |
| 02 | `02_clean_who.py` | Clean WHO suicide data |
| 03 | `03_clean_gbd.py` | Clean IHME GBD data |
| 04 | `04_merge_ml.py` | Merge into the modeling table |
| 05 | `05_merge_context.py` | Add context tables |
| 06 | `06_ml_baseline.py` | Baseline ML model |
| 07 | `07_data_quality_scorecard.py` | Data-quality scorecard |
| 08 | `08_segmentation_outliers.py` | KMeans segmentation + outliers |
| 09 | `09_source_agreement.py` | WHO vs IHME cross-source check |
| 10 | `10_enrich_covariates.py` | Fetch World Bank covariates |
| 11 | `11_ml_enriched.py` | Leakage-free enriched model |

## v1 advanced suite (real data)

`12_ml_advanced.py` (LightGBM + Optuna + SHAP + conformal) ·
`13_hierarchical_model.py` (mixed-effects + ICC) ·
`14_umap_embedding.py` · `15_country_network.py` ·
`16_subgroup_rules.py` (subgroup discovery + association rules)

## v2 / v3

`v2_*.py` — advanced-analytics methods showcase on synthetic data
(clustering, DTW, graphs, forecasting, quantile, explainability, validation).
`v3_prepare_features.py` — features for the interactive risk estimator.

## Shared / app

| File | Purpose |
| --- | --- |
| `project_paths.py` | Version-aware paths (via `MHP_VERSION`) |
| `advanced_common.py` | Shared enriched-data loader |
| `theme.py` | Design tokens + Plotly/matplotlib styling |
| `app.py` | Streamlit dashboard (all versions) |
| `v0_visuals.py` | v0 static visual gallery |

## Running

Prefer the one-command runners in `scripts/` (`run_v1_pipeline.py`,
`run_advanced.py`, `run_v2_pipeline.py`, `run_v3_pipeline.py`). Each script sets
`MHP_VERSION` so `project_paths` writes to the right `v*/` folder.
