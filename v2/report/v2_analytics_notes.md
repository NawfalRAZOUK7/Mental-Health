# v2 Analytics Notes

- Clustering: KMeans with k=3 selected by silhouette.
- Outliers: IsolationForest (contamination=0.05).
- Forecast: linear regression by region (2024-2030).
- Scenario model: linear regression on standardized features (2010+).

Outputs:
- v2/data_clean/v2_clusters.csv
- v2/report/v2_cluster_centers.csv
- v2/report/v2_k_selection.csv
- v2/report/v2_outliers.csv
- v2/report/v2_forecast_region.csv
- v2/report/v2_model_coeffs.csv
- v2/report/v2_model_metrics.csv