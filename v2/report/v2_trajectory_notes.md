# v2 Trajectory Clustering Notes

Clustering uses trajectory features from 2000-2023 suicide_rate time series.
- Selected k=3 by silhouette score.

Features:
- slope (linear trend)
- volatility (std dev)
- peak_value and peak_year
- last5_change (last 5y mean - long-run mean)
- mean_rate (overall mean)

Outputs:
- v2/data_clean/v2_trajectory_clusters.csv
- v2/report/v2_trajectory_cluster_centers.csv
- v2/report/v2_trajectory_k_selection.csv