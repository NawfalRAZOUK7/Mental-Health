# Data Quality Scorecard

## Dataset overview
| dataset | rows | columns | year_min | year_max | iso3_missing_count | iso3_missing_pct | duplicate_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| context_allcauses_trend | 5589 | 12 | 2021 | 2023 | 189 | 3.38% | 18 |
| context_big_categories_2023 | 29772 | 13 | 2023 | 2023 | 29382 | 98.69% | 1625 |
| context_probdeath_2023 | 615 | 13 | 2023 | 2023 | 21 | 3.41% | 0 |
| gbd_addiction_clean | 2355 | 17 | 2023 | 2023 | 1731 | 73.50% | 0 |
| gbd_allcauses_clean | 63585 | 17 | 2021 | 2023 | 46737 | 73.50% | 0 |
| gbd_big_categories_clean | 51480 | 19 | 2023 | 2023 | 50700 | 98.48% | 0 |
| gbd_depression_dalys_clean | 1836 | 17 | 2023 | 2023 | 0 | 0.00% | 0 |
| gbd_prob_death_clean | 93480 | 17 | 2023 | 2023 | 456 | 0.49% | 0 |
| gbd_selfharm_clean | 1224 | 17 | 2023 | 2023 | 0 | 0.00% | 0 |
| merged_ml_country | 549 | 17 | 2021 | 2021 | 0 | 0.00% | 0 |
| ml_baseline_features | 183 | 9 | n/a | n/a | 0 | 0.00% | 0 |
| who_2021_clean | 549 | 10 | 2021 | 2021 | 0 | 0.00% | 0 |

## ISO3 unmatched (by source_type)
| source_type | count |
| --- | --- |
| GBD | 574 |

## WHO data_quality distribution
| data_quality | count | pct |
| --- | --- | --- |
| Very low | 225 | 40.98 |
| High | 174 | 31.69 |
| Medium | 84 | 15.3 |
| Low | 66 | 12.02 |

## Missingness (top 12 columns)
| dataset | column | missing_count | missing_pct |
| --- | --- | --- | --- |
| gbd_prob_death_clean | upper | 93480 | 100.0 |
| gbd_prob_death_clean | lower | 93480 | 100.0 |
| context_big_categories_2023 | iso3 | 29382 | 98.69 |
| gbd_big_categories_clean | iso3 | 50700 | 98.48 |
| context_probdeath_2023 | upper | 594 | 96.59 |
| context_probdeath_2023 | lower | 594 | 96.59 |
| gbd_allcauses_clean | iso3 | 46737 | 73.5 |
| gbd_addiction_clean | iso3 | 1731 | 73.5 |
| context_probdeath_2023 | region_name | 54 | 8.78 |
| context_allcauses_trend | region_name | 486 | 8.7 |
| context_probdeath_2023 | iso3 | 21 | 3.41 |
| context_allcauses_trend | iso3 | 189 | 3.38 |

## Files
- v1/report/data_quality_scorecard.csv
- v1/report/data_quality_missingness.csv
- v1/report/data_quality_who_data_quality.csv
- v1/report/data_quality_iso3_unmatched.csv