# Pattern Mining (real data): subgroups + association rules

Target: high suicide rate (top tercile). Base rate: 33% of countries.

## Top subgroups (share of high-suicide countries far above base rate)
- region_name=='African Region' (quality 0.051, n=47, share 53%)
- life_expectancy_years<66.60 AND region_name=='African Region' (quality 0.044, n=30, share 60%)
- region_name=='African Region' AND urban_population_pct: [37.21:56.08[ (quality 0.040, n=17, share 76%)
- life_expectancy_years<66.60 (quality 0.038, n=36, share 53%)
- life_expectancy_years<66.60 AND urban_population_pct: [37.21:56.08[ (quality 0.038, n=15, share 80%)

## Top association rules => high suicide (lift > 1.2)
- addiction_death_rate=high + alcohol_litres_per_capita=high + depression_dalys_rate=low => high suicide (conf 0.71, lift 2.14)
- addiction_death_rate=high + depression_dalys_rate=low + region_name=European Region => high suicide (conf 0.67, lift 2.00)
- health_exp_per_capita_usd=low + income_group=LI + life_expectancy_years=low + region_name=African Region + urban_population_pct=low => high suicide (conf 0.67, lift 2.00)
- income_group=LI + life_expectancy_years=low + region_name=African Region + urban_population_pct=low => high suicide (conf 0.67, lift 2.00)
- health_exp_per_capita_usd=low + income_group=LI + region_name=African Region + urban_population_pct=low => high suicide (conf 0.67, lift 2.00)
- income_group=LI + region_name=African Region + urban_population_pct=low => high suicide (conf 0.67, lift 2.00)

## Outputs
- subgroups.csv, assoc_rules_real.csv