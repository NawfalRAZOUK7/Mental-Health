# Dataset Usage Matrix

| Dataset | Used in dashboard page | Used in merged ML table? | Why/why not |
| --- | --- | --- | --- |
| data_raw/IHME-GBD_2023_DATA-age-standardized-death-rate-1.csv | GBD Addiction (Deaths rate) | yes | Core 4 for ML; deaths rate for addiction-focused causes. |
| data_raw/IHME-GBD_2023_DATA-all-cause-burden-all-ages-1.csv | Extra tab: All-cause context trends (648b4e83) | no | Context-only trend view; excluded from ML to keep one consistent table. |
| data_raw/IHME-GBD_2023_DATA-anemia-prevalence-ylds-1.csv | Not used (alternate to all-cause context trends) | no | Optional swap for 648b4e83; not used in current tab plan. |
| data_raw/IHME-GBD_2023_DATA-dalys-causes-1.csv | GBD Depression (DALYs) | yes | Core 4 for ML; DALYs view for depression-related causes. |
| data_raw/IHME-GBD_2023_DATA-deaths-mental-substance-violence-1.csv | GBD Self-harm (Deaths rate) | yes | Core 4 for ML; deaths rate for self-harm causes. |
| data_raw/IHME-GBD_2023_DATA-probability-of-death-1.csv | Extra tab: Probability of death (a8aba8d1) | no | Extra explorer tab only. |
| data_raw/IHME-GBD_2023_DATA-risk-factor-burden-1.csv | Extra tab: Big categories (376b1818) | no | Extra explorer tab only. |
| data_raw/who_africa_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
| data_raw/who_americas_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
| data_raw/who_emro_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
| data_raw/who_europe_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
| data_raw/who_global_master.csv | WHO Suicide tab (global) | yes | Core 4 for ML; global suicide baseline. |
| data_raw/who_searo_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
| data_raw/who_wpro_region_full.csv | WHO Suicide tab (regional) | no | Optional regional drilldown; excluded from ML for consistency. |
