# v2 Synthetic Validity Report

Comparison of v1 (real) vs v2 (synthetic) distributions for 2021 (Both sexes).

| feature | v1_mean | v1_median | v1_p10 | v1_p90 | v2_mean | v2_median | v2_p10 | v2_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| suicide_rate | 9.11 | 8.30 | 2.10 | 16.76 | 98.21 | 96.14 | 66.92 | 129.09 |
| depression_dalys_rate | 704.10 | 672.48 | 473.69 | 971.65 | 730.79 | 717.73 | 483.63 | 996.30 |
| addiction_death_rate | 3.85 | 2.26 | 0.89 | 8.35 | 3.97 | 2.28 | 0.87 | 8.45 |
| selfharm_death_rate | 8.62 | 8.66 | 3.00 | 14.05 | 10.02 | 9.60 | 3.19 | 16.91 |

## Distribution Tests

| feature | ks_stat | ks_pvalue | wasserstein |
| --- | --- | --- | --- |
| suicide_rate | 1.0000 | 0.0000 | 89.0933 |
| depression_dalys_rate | 0.1475 | 0.0371 | 33.0899 |
| addiction_death_rate | 0.0328 | 1.0000 | 0.1405 |
| selfharm_death_rate | 0.1639 | 0.0145 | 1.4025 |

Notes:
- v1 values are averaged across age groups per country.
- v2 values are synthetic and intended for demonstration only.
- Synthetic rates may differ in scale from v1; interpret patterns over magnitudes.
- KS/Wasserstein tests included via scipy.
