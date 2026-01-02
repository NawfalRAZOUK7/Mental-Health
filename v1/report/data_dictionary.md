# Data Dictionary

## WHO Suicide (v1/data_clean/who_2021_clean.csv)
| Column | Definition | Unit | Notes |
| --- | --- | --- | --- |
| region_name | WHO region | n/a | WHO region grouping |
| location_name | Country name | n/a | WHO naming |
| iso3 | ISO3 country code | n/a | Missing for non-country rows |
| sex_name | Sex category | n/a | Male, Female, Both sexes |
| year | Report year | year | 2021 |
| number_suicides_2021 | Total suicides | count | WHO |
| crude_suicide_rate_2021 | Crude suicide rate | per 100k | WHO |
| age_standardized_suicide_rate_2021 | Age-standardized suicide rate | per 100k | WHO |
| data_quality | WHO data quality label | n/a | High/Medium/Low/Very low |
| income_group | Income group | n/a | World Bank classes |

## GBD Clean Files (v1/data_clean/gbd_*_clean.csv)
Applies to: gbd_depression_dalys_clean, gbd_addiction_clean, gbd_selfharm_clean, gbd_allcauses_clean, gbd_prob_death_clean, gbd_big_categories_clean.

| Column | Definition | Unit | Notes |
| --- | --- | --- | --- |
| measure_name | Health measure | n/a | DALYs, Deaths, or Probability of death (prob_death file) |
| metric_name | Metric type | n/a | Number/Percent/Rate for DALYs or Deaths; Probability of death for prob_death |
| location_name | Country or aggregate name | n/a | GBD naming |
| iso3 | ISO3 country code | n/a | Missing for aggregates/subnationals |
| sex_name | Sex category | n/a | Male, Female, Both |
| age_name | Age group label | n/a | GBD age groups |
| cause_name | Cause name | n/a | GBD cause taxonomy |
| year | Report year | year | Often 2023 or multi-year |
| val | Estimate value | varies | Rate per 100k, Number, Percent, or Probability (0-1) |
| upper | Upper uncertainty interval | same as val | 95% UI |
| lower | Lower uncertainty interval | same as val | 95% UI |

Additional columns for gbd_big_categories_clean:
| Column | Definition | Unit | Notes |
| --- | --- | --- | --- |
| rei_id | Risk exposure ID | n/a | Only in big categories file |
| rei_name | Risk exposure name | n/a | Only in big categories file |

## Merged ML Dataset (v1/data_clean/merged_ml_country.csv)
| Column | Definition | Unit | Notes |
| --- | --- | --- | --- |
| iso3 | ISO3 country code | n/a | Primary join key |
| location_name | Country name | n/a | WHO naming |
| region_name | WHO region | n/a | WHO regions |
| year | WHO year | year | 2021 |
| number_suicides_2021 | Total suicides | count | WHO |
| crude_suicide_rate_2021 | Crude suicide rate | per 100k | WHO |
| age_standardized_suicide_rate_2021 | Age-standardized suicide rate | per 100k | WHO |
| data_quality | WHO data quality | n/a | High/Medium/Low/Very low |
| income_group | Income group | n/a | World Bank classes |
| age_name | Age group label | n/a | GBD age groups |
| gbd_addiction_death_rate_both | Addiction deaths rate | per 100k | GBD 2023 |
| gbd_addiction_death_rate_female | Addiction deaths rate (female) | per 100k | GBD 2023 |
| gbd_addiction_death_rate_male | Addiction deaths rate (male) | per 100k | GBD 2023 |
| gbd_depression_dalys_rate_both | Depression DALYs rate | per 100k | GBD 2023 |
| gbd_selfharm_death_rate_female | Self-harm deaths rate (female) | per 100k | GBD 2023 |
| gbd_selfharm_death_rate_male | Self-harm deaths rate (male) | per 100k | GBD 2023 |
| gbd_year | GBD year | year | 2023 |

## Context Tables (v1/data_clean/context_tables/*.csv)
Applies to: context_allcauses_trend, context_big_categories_2023, context_probdeath_2023.

| Column | Definition | Unit | Notes |
| --- | --- | --- | --- |
| location_type | Geographic level | n/a | country, who_region, global, gbd_aggregate |
| region_name | WHO region | n/a | For country/region rows |
| iso3 | ISO3 country code | n/a | Missing for aggregates |
| location_name | Location name | n/a | Country or aggregate |
| sex_name | Sex category | n/a | Male, Female, Both |
| age_name | Age group label | n/a | GBD age groups |
| year | Report year | year | All-cause includes multi-year |
| cause_name | Cause name | n/a | Only for big categories / prob death |
| measure_name | Health measure | n/a | DALYs (all-cause/big categories) or Probability of death |
| metric_name | Metric type | n/a | Number/Percent/Rate for DALYs; Probability of death for prob_death |
| val | Estimate value | varies | Rate per 100k, Number, Percent, or Probability (0-1) |
| upper | Upper uncertainty interval | same as val | 95% UI |
| lower | Lower uncertainty interval | same as val | 95% UI |
