# v2 — Advanced Analytics (Methods Showcase)

> **Read this first.** v2 is a **techniques showcase on synthetic data**, not a source of
> real-world findings. The synthetic tables are generated from v1 distributions, so any
> "pattern" recovered here reflects the generator, not reality. The value of v2 is
> demonstrating that the methods are implemented correctly end-to-end — **capability, not insight.**
> For real-data results, see v1.

## Techniques demonstrated

| Area | Method | Output |
| --- | --- | --- |
| Segmentation | KMeans + silhouette k-selection | `v2_clusters.csv`, `v2_cluster_centers.csv` |
| Trajectory clustering | DTW clustering (tslearn) | `v2_dtw_clusters.csv`, `v2_trajectory_clusters.csv` |
| Graph analysis | Similarity graph + greedy-modularity communities (networkx) | `v2_graph_clusters.csv`, `v2_graph_centrality.csv` |
| Association mining | Apriori rules on binned indicators (mlxtend) | `v2_assoc_rules.csv` |
| Forecasting (baseline) | Expanding-window AR(5) linear backtest | `v2_backtest_metrics.csv` |
| Forecasting (deep) | GRU/LSTM (PyTorch) | `v2_dl_metrics.csv`, `v2_dl_forecast_region.csv` |
| Uncertainty | Quantile gradient boosting + pinball loss + coverage | `v2_quantile_metrics.csv` |
| Explainability | Permutation importance + partial dependence | `v2_perm_importance.csv`, `v2_partial_dependence.csv` |
| Anomalies | IsolationForest outliers | `v2_outliers.csv` |
| Change detection | Change-point detection | `v2_changepoints.csv` |
| Data validation | Great Expectations suite | `ge_report.html`, `v2_validity_report.md` |

## Honest baseline-vs-deep note

The deep forecaster (GRU/LSTM) and the simple AR(5) linear backtest are both included on
purpose. The series are short (~24 points per region), so a neural network is **not expected
to beat** a well-specified linear baseline here — the comparison is kept to avoid overselling
the deep model. Treat both as methodology demos, not validated forecasts.

## Contents
- `v2/data_clean`: synthetic tables + cluster outputs
- `v2/report`: analytics outputs, notes, validation, forecasts
- `v2/notebooks`: v2 analysis + modeling notebooks
- `v2/assets`: optional exported figures

## Rebuild
```bash
pip install -r requirements.txt
pip install -r requirements-v2.txt
python scripts/run_v2_pipeline.py
```

## Run
```bash
python scripts/run_app.py --version v2
```
