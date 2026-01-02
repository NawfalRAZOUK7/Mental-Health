# Context Table Metadata

## context_allcauses_trend.csv

- Purpose: Macro DALYs trend view across countries, WHO regions, and global.
- Source: v1/data_clean/gbd_allcauses_clean.csv
- Filters: cause_name = All causes; measure_name = DALYs (Disability-Adjusted Life Years); metric_name in Number/Percent/Rate; country rows kept by ISO3 + preferred location-name list (WHO names + overrides + exact pycountry matches).
- Aggregation:
  - Country rows retained (all sexes, all ages, years 2021-2023).
  - WHO-region and global rows computed from countries using population-weighted aggregation:
    - Number: summed
    - Rate: weighted by population (population derived from Number/Rate)
    - Percent: weighted by Number
- Columns: location_type (country|who_region|global), region_name, iso3, location_name, sex_name, age_name, year, measure_name, metric_name, val, upper, lower.

## context_big_categories_2023.csv

- Purpose: 2023 big-category DALYs overview using GBD aggregate locations.
- Source: v1/data_clean/gbd_big_categories_clean.csv
- Filters: measure_name = DALYs (Disability-Adjusted Life Years); metric_name in Number/Percent/Rate; year = 2023.
- Aggregation:
  - Locations are already GBD aggregates (regions, SDI groups, etc.), not country-level.
  - Both sexes derived from Male/Female within each location using population weights (Number/Rate).
  - No WHO-region/global recomputation to avoid double-counting across overlapping aggregates.
- Columns: location_type = gbd_aggregate, region_name (same as location_name), iso3 (blank for aggregates), location_name, sex_name, age_name, year, cause_name, measure_name, metric_name, val, upper, lower.

## context_probdeath_2023.csv

- Purpose: 2023 probability-of-death snapshot by country, WHO region, and global.
- Source: v1/data_clean/gbd_prob_death_clean.csv
- Filters: year = 2023; age_name = All ages; cause_name = All causes; country rows kept by ISO3 + preferred location-name list (WHO names + overrides + exact pycountry matches).
- Aggregation:
  - Country rows retained (all sexes).
  - WHO-region and global rows computed from countries using population-weighted averages.
  - Population weights come from all-cause DALYs Number/Rate (context_allcauses_trend).
- Columns: location_type (country|who_region|global), region_name, iso3, location_name, sex_name, age_name, year, cause_name, measure_name, metric_name, val, upper, lower.
