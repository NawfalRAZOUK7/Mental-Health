# Merge Quality Report

## Dataset A — Merged ML (v1/data_clean/merged_ml_country.csv)

- WHO 2021 backbone (Both sexes): 183 rows
- After ISO3 filter: 183 rows
- Final merged rows: 549 (3 age groups per country)
- Missingness (GBD features): 0% across all sex-specific columns

GBD feature filters and aggregation:
- Depression: `Depressive disorders`, `DALYs (Disability-Adjusted Life Years)`, `Rate`, `Both`, year 2023; age groups kept separate (<20, 20-24, 25+).
- Addiction: `Substance use disorders`, `Deaths`, `Rate`, year 2023; sex-specific columns (Male/Female/Both) at age-standardized level, repeated across age groups in ML table.
- Self-harm: `Self-harm`, `Deaths`, `Rate`, year 2023; sex-specific columns (Male/Female) with age groups kept separate (<20, 20-24, 25+).

Cross-year merge note:
- WHO is 2021 only; GBD features are 2023. This is a cross-year merge, so ML uses WHO suicide outcomes from 2021 paired with 2023 GBD features.

## Dataset B — Context Tables (v1/data_clean/context_tables/)

- context_allcauses_trend.csv: 5,589 rows (All causes, DALYs, metrics Number/Percent/Rate, all sexes; years 2021-2023; includes WHO-region and global aggregates)
- context_big_categories_2023.csv: 29,772 rows (DALYs, metrics Number/Percent/Rate, year 2023; Male/Female plus derived Both; GBD aggregate locations only)
- context_probdeath_2023.csv: 615 rows (All causes, Probability of death, all sexes, All ages; year 2023; includes WHO-region and global aggregates)

Dropped/filtered items:
- Non-country aggregates from GBD are excluded for all-cause and probability-of-death tables; region/global values are recomputed from countries using WHO region mapping (population-weighted from Number/Rate-derived population).
- GBD years other than 2023 are excluded for ML and context tables except all-cause trends (2021-2023).
- All-cause context trends use DALYs only (Deaths/YLLs excluded); probability-of-death context restricted to All causes + All ages.
- Big-categories table keeps GBD aggregate locations as provided (no recomputed WHO regions/global because categories are already aggregated).
- Causes restricted to `Depressive disorders`, `Substance use disorders`, and `Self-harm` for ML features.
