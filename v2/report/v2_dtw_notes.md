# v2 DTW Clustering Notes

DTW clustering on 2000-2023 suicide_rate time series (Both sexes).
- Selected k=5.

Preprocessing:
- Per-country interpolation over years to fill missing values.
- Per-series z-score standardization before DTW.

Outputs:
- v2/report/v2_dtw_clusters.csv
- v2/report/v2_dtw_cluster_centers.csv
- v2/report/v2_dtw_k_selection.csv