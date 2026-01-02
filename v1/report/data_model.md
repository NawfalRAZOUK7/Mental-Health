# Data Model (Star Schema)

## Grain
- Fact table grain: Country x Age Group x Year (WHO year; GBD year stored separately).
- Sex-specific GBD rates are stored as separate measures (male/female) in the ML fact.

## Core Fact Table
FACT_MentalHealth_CountryYearAge (`v1/data_clean/merged_ml_country.csv`)

Measures:
- number_suicides_2021 (count)
- crude_suicide_rate_2021 (per 100k)
- age_standardized_suicide_rate_2021 (per 100k)
- gbd_depression_dalys_rate_both (DALYs rate per 100k)
- gbd_addiction_death_rate_both/female/male (deaths rate per 100k)
- gbd_selfharm_death_rate_female/male (deaths rate per 100k)

## Dimensions
- DIM_Country: iso3, location_name, region_name, income_group
- DIM_Time: year (WHO), gbd_year (GBD)
- DIM_AgeGroup: age_name
- DIM_Sex: sex_name in source tables; represented as separate measures in the ML fact

## Context Fact Tables (Analytics Layer)
- FACT_Context_AllCause (`v1/data_clean/context_tables/context_allcauses_trend.csv`)
- FACT_Context_ProbDeath (`v1/data_clean/context_tables/context_probdeath_2023.csv`)
- FACT_Context_BigCategories (`v1/data_clean/context_tables/context_big_categories_2023.csv`)

## Diagram (ASCII)
```
            DIM_Time
               |
DIM_Country -- FACT_MentalHealth_CountryYearAge -- DIM_AgeGroup
               |
           DIM_Sex (source tables)
```
