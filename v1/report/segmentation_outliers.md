# Segmentation + Outliers Notes

## Features
- age_standardized_suicide_rate_2021 (WHO)
- gbd_depression_dalys_rate_both (GBD 2023)
- gbd_addiction_death_rate_both (GBD 2023, substance use disorders)
- gbd_selfharm_death_rate_both (GBD 2023, mean of male/female)

## Clustering
- KMeans with k selected from [3, 4, 5, 6] by max silhouette.
- Selected k = 3.

## Outliers
- Z-score threshold: 2.5
- Outlier reason lists which features are high/low vs population.

## Outputs
- v1/data_clean/segmentation_features.csv
- v1/data_clean/segmentation_clusters.csv
- v1/report/segmentation_cluster_centers.csv
- v1/report/segmentation_k_selection.csv
- v1/report/outliers_table.csv