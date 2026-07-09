# Report Plan (temporary)

## 0) Front matter
- Title
- Authors, affiliation, date
- Abstract (150 to 250 words)
- Keywords (5 to 7)

## 1) Introduction
- Motivation: global mental health burden
- Why suicide and mental disorders matter for public health
- Scope of the project and target audience

## 2) Objectives and research questions
- O1: Describe global and regional suicide patterns
- O2: Compare related burdens (depression, addiction, self-harm)
- O3: Explore relationships and build baseline predictive models
- O4: Demonstrate BI and data mining techniques
- O5: Provide reproducible pipelines and versions

## 3) Data sources and coverage
- Table: dataset catalog (source, year, metric, measure, file id, used in version)
- WHO 2021 suicide dataset (global + regions)
- GBD datasets used in v1 (core)
- GBD datasets used for context (extra)
- Additional GBD datasets used in v0 static visuals
- Notes on time coverage and units

## 4) Data preparation and pipeline
- Inventory and profiling (src/00_inventory.py)
- Country mapping and ISO3 matching (src/01_country_mapping.py)
- WHO cleaning (src/02_clean_who.py)
- GBD cleaning per file (src/03_clean_gbd.py)
- Merges:
  - ML dataset (src/04_merge_ml.py)
  - Context tables (src/05_merge_context.py)
- ML baseline (src/06_ml_baseline.py)
- Versioned pipelines (scripts/run_v1_pipeline.py, scripts/run_v2_pipeline.py, scripts/run_v3_pipeline.py)

## 5) BI data model and KPIs
- Star schema: fact table and dimensions
- Grain definition (Country x Year x Sex)
- KPI definitions and units
- Data dictionary reference (report/data_dictionary.md)

## 6) Versioned builds (what each version demonstrates)
- v0: static visuals, maximum chart variety, minimal transforms
- v1: core dashboard on real data
- v2: synthetic advanced analytics and BI/data mining extras
- v3: risk estimator and user inputs
- Table: version features and outputs

## 7) Results and dashboards (v1 core)
- Overview page (KPI cards, regional trends)
- WHO Suicide Explorer (map, rankings, sex comparison)
- Depression DALYs (map, top 20, age breakdown)
- Addiction (cause dropdown, map, sex comparison)
- Self-harm (map, sex comparison, optional methods)
- Probability of death (map + ranking)
- All-cause trends (time series by metric)
- Big categories (treemap or stacked view)
- Relationships (scatter + correlation)
- ML demo (Ridge, RandomForest, metrics)
- Methods and limitations page

## 8) Advanced analytics (v2)
- Clustering and segmentation
- Outliers and alerts
- Forecasts and backtest
- Quantile regression and prediction intervals
- Explainability (permutation importance, partial dependence)
- Graph clustering and association rules
- Scenario lab (what-if)
- Data quality summary and GE report

## 9) Risk estimator (v3)
- Problem framing and cutoff definition
- Model inputs and outputs
- Calibration and reliability
- Counterfactual hints and feature drivers

## 10) Data quality and limitations
- Missingness and unmatched ISO3
- Cross-year merge caveats (WHO 2021 vs GBD 2023)
- Ecological fallacy risks
- Synthetic data limitations (v2/v3)

## 11) Reproducibility
- Requirements files (core and v2)
- How to run each version (refer to VERSIONS.md)
- Expected outputs and folders

## 12) Conclusion and future work
- Key findings summary
- Potential extensions (more data, methods, validation)

## Appendices
- A: Dataset catalog (dataset_catalog.csv)
- B: Usage matrix (dataset_usage_matrix.md)
- C: Merge quality report (merge_quality.md)
- D: Tables and figures list with filenames
