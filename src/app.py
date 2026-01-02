#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import base64
import html as html_lib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from project_paths import DATA_CLEAN, REPORT_DIR, VERSION

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    ColumnTransformer = None  # type: ignore
    LogisticRegression = None  # type: ignore
    NearestNeighbors = None  # type: ignore
    accuracy_score = None  # type: ignore
    roc_auc_score = None  # type: ignore
    brier_score_loss = None  # type: ignore
    CalibratedClassifierCV = None  # type: ignore
    OneHotEncoder = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore


CONTEXT_DIR = DATA_CLEAN / "context_tables"
V3_NUMERIC_COLS = [
    "depression_dalys_rate",
    "addiction_death_rate",
    "selfharm_death_rate",
]


st.set_page_config(page_title="Mental Health Dashboard", layout="wide")


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:wght@400;600&display=swap');

:root {
  --bg-1: #f4efe7;
  --bg-2: #f7f2ea;
  --ink: #1c1b1a;
  --muted: #6b6460;
  --accent: #1f6f8b;
  --accent-2: #f2b950;
  --card: #ffffff;
  --border: #e4ddd4;
}

html, body, [class*="css"]  {
  font-family: "Space Grotesk", system-ui, sans-serif;
  color: var(--ink);
}

.stApp {
  background: radial-gradient(circle at top left, var(--bg-2), var(--bg-1));
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 12px 24px rgba(28, 27, 26, 0.05);
}

.kpi-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.kpi-value {
  font-size: 26px;
  font-weight: 600;
}

.section-title {
  font-family: "Source Serif 4", serif;
  font-size: 26px;
  font-weight: 600;
  margin-bottom: 4px;
}

.section-subtitle {
  color: var(--muted);
  margin-bottom: 18px;
}

.info-chip {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(31, 111, 139, 0.12);
  color: var(--accent);
  font-size: 12px;
}

.guide-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  margin: 14px 0 18px 0;
  box-shadow: 0 10px 22px rgba(28, 27, 26, 0.06);
}
.guide-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 6px;
}
.guide-line {
  margin: 3px 0;
  color: var(--muted);
  font-size: 13px;
}
.chart-guide {
  margin: 6px 0 12px 0;
  font-size: 12px;
  color: var(--muted);
}
.chart-guide strong { color: var(--ink); }
</style>
""",
    unsafe_allow_html=True,
)


V1_PAGE_GUIDES = {
    "overview": {
        "objective": "Summarize the 2021 suicide burden and highlight key regional patterns.",
        "how": "Use the KPIs for scale, then compare the top countries and regional spread.",
        "notes": "All metrics are rates per 100k from WHO 2021; the correlation uses 25+ years.",
    },
    "who": {
        "objective": "Explore WHO 2021 suicide rates by sex and metric.",
        "how": "Pick sex and metric; the map and ranking update. Use the scatter to compare crude vs age-standardized.",
        "notes": "Age-standardized rates control for age structure, crude rates do not.",
    },
    "depression": {
        "objective": "Compare depression DALYs rates across countries and age groups.",
        "how": "Filter age for the map and top-20 chart; the summary bar uses all ages (Both sexes).",
        "notes": "DALYs combine years of life lost and years lived with disability.",
    },
    "addiction": {
        "objective": "Compare substance-use death rates by cause and sex.",
        "how": "Select cause and sex to update the map and rankings, then compare sexes for a country.",
        "notes": "Rates are per 100k and use 2023 GBD values.",
    },
    "selfharm": {
        "objective": "View self-harm death rates by age group and sex.",
        "how": "Choose age and sex for the map and rankings, then compare sexes for a country.",
        "notes": "Self-harm is filtered as a single cause in the GBD file.",
    },
    "prob_death": {
        "objective": "Show probabilities of death, which are not rates.",
        "how": "Filter cause, sex, and age. Use the map for geographic spread and the bar chart for ranking.",
        "notes": "Values are probabilities and should not be compared directly to per-100k rates.",
    },
    "allcause": {
        "objective": "Track all-cause trends over time for countries, regions, and global aggregates.",
        "how": "Select location type, location, sex, age, and metric to view the trend line.",
        "notes": "Metrics include Number and Rate depending on availability.",
    },
    "big_categories": {
        "objective": "Show the composition of DALYs by broad cause groups.",
        "how": "Use the treemap for hierarchy and the donut for top-level shares.",
        "notes": "Treemap rolls up cause levels; donut shows Level 1 only.",
    },
    "relationships": {
        "objective": "Inspect relationships between suicide rates and GBD indicators.",
        "how": "Pick variables for the scatter, then review the correlation heatmap.",
        "notes": "These are ecological correlations, not causal effects.",
    },
    "segmentation": {
        "objective": "Cluster countries into profiles based on mental-health indicators.",
        "how": "Use the map for spatial patterns and the profile chart for cluster signatures.",
        "notes": "Clusters are unsupervised and meant for descriptive grouping.",
    },
    "outliers": {
        "objective": "Surface countries with unusual indicator combinations.",
        "how": "Select two indicators and review the outlier flags and scores.",
        "notes": "Outliers are statistical, not necessarily errors.",
    },
    "ml_demo": {
        "objective": "Compare two baseline models for suicide-rate prediction.",
        "how": "Review the holdout metrics, cross-validation table, and feature importance.",
        "notes": "One row per country is used to avoid leakage across age groups.",
    },
    "methods": {
        "objective": "Document sources, definitions, and data quality checks.",
        "how": "Use this page to understand how the clean tables were constructed.",
        "notes": "See report outputs for data quality and modeling details.",
    },
}

V1_CHART_GUIDES = {
    "overview_kpis": {
        "title": "Overview KPIs",
        "objective": "Provide a quick scale check for suicide rates and coverage.",
        "how": "KPIs show country count, mean, and median for both sexes in 2021.",
        "why": "KPIs establish the baseline before exploring distributions.",
    },
    "overview_top10": {
        "title": "Top 10 suicide rates",
        "objective": "Highlight the countries with the highest age-standardized rates.",
        "how": "Bars are ordered by rate; longer bars indicate higher rates.",
        "why": "Ranked views make extremes easy to compare.",
    },
    "overview_region_box": {
        "title": "Regional spread",
        "objective": "Compare regional distributions, not just averages.",
        "how": "Boxes show medians and spread; dots represent countries.",
        "why": "Boxplots reveal variability within regions.",
    },
    "overview_corr": {
        "title": "Model demo signal",
        "objective": "Check whether suicide and depression move together in the merged table.",
        "how": "Correlation is computed for 25+ years, both sexes.",
        "why": "It provides a quick sanity check before modeling.",
    },
    "who_map": {
        "title": "WHO map",
        "objective": "Show geographic variation in the selected metric.",
        "how": "Darker color means higher rate; hover for exact values.",
        "why": "Maps surface spatial clustering quickly.",
    },
    "who_top_bottom": {
        "title": "Top and bottom countries",
        "objective": "Rank the extremes for the selected metric.",
        "how": "Left bars show top 10, right bars show bottom 10.",
        "why": "Side-by-side ranks show dispersion.",
    },
    "who_sex_compare": {
        "title": "Sex comparison",
        "objective": "Compare male vs female rates for a country.",
        "how": "Bars show the selected metric for each sex.",
        "why": "Sex gaps are a key epidemiologic pattern.",
    },
    "who_crude_vs_age": {
        "title": "Crude vs age-standardized",
        "objective": "Show how age structure changes the rate comparison.",
        "how": "Points above the diagonal mean age-standardized is higher.",
        "why": "It explains why crude and age-standardized rates can differ.",
    },
    "depression_map": {
        "title": "Depression DALYs map",
        "objective": "Map DALYs rate across countries for the selected age group.",
        "how": "Color encodes DALYs rate; hover for exact values.",
        "why": "Maps show where burden concentrates.",
    },
    "depression_top20": {
        "title": "Top 20 DALYs",
        "objective": "Rank the highest-burden countries for the selected age group.",
        "how": "Bars are ordered by DALYs rate.",
        "why": "Rankings make outliers visible.",
    },
    "depression_age_bar": {
        "title": "DALYs by age group",
        "objective": "Compare average DALYs rates across age groups.",
        "how": "Bars show the mean for each age bucket, both sexes.",
        "why": "Highlights which age group carries the highest burden.",
    },
    "addiction_map": {
        "title": "Addiction map",
        "objective": "Map deaths rate for the selected substance-use cause and sex.",
        "how": "Color encodes deaths rate; hover for exact values.",
        "why": "Spatial patterns differ by substance.",
    },
    "addiction_top20": {
        "title": "Top 20 addiction deaths",
        "objective": "Rank the highest-burden countries for the selected cause and sex.",
        "how": "Bars are ordered by deaths rate.",
        "why": "Rankings clarify the distribution tail.",
    },
    "addiction_sex_compare": {
        "title": "Sex comparison",
        "objective": "Compare male vs female deaths rates for a country.",
        "how": "Bars show the average for the selected cause.",
        "why": "Sex differences are often large in substance use.",
    },
    "selfharm_map": {
        "title": "Self-harm map",
        "objective": "Map self-harm deaths rate for the selected age and sex.",
        "how": "Color encodes deaths rate; hover for exact values.",
        "why": "Maps show geographic variation quickly.",
    },
    "selfharm_top20": {
        "title": "Top 20 self-harm deaths",
        "objective": "Rank the highest-burden countries for the selected age and sex.",
        "how": "Bars are ordered by deaths rate.",
        "why": "Rankings reveal extremes.",
    },
    "selfharm_sex_compare": {
        "title": "Sex comparison",
        "objective": "Compare male vs female self-harm rates for a country.",
        "how": "Bars show the selected age group for each sex.",
        "why": "Sex gaps indicate different risk profiles.",
    },
    "selfharm_methods": {
        "title": "Methods breakdown",
        "objective": "Compare average rates by method, if available.",
        "how": "Bars show average rates by method category.",
        "why": "Methods show how burden differs within self-harm.",
    },
    "probdeath_map": {
        "title": "Probability of death map",
        "objective": "Map probability of death for the selected cause, sex, and age.",
        "how": "Color encodes probability; values are not per-100k rates.",
        "why": "Probabilities show relative risk levels.",
    },
    "probdeath_top20": {
        "title": "Top 20 probabilities",
        "objective": "Rank countries with highest probability of death.",
        "how": "Bars are ordered by probability value.",
        "why": "Ranking shows the highest-risk countries.",
    },
    "allcause_trend": {
        "title": "All-cause trend line",
        "objective": "Track changes over time for the selected metric.",
        "how": "Line shows the metric value by year.",
        "why": "Trends show direction and magnitude of change.",
    },
    "bigcat_treemap": {
        "title": "Big categories treemap",
        "objective": "Show how DALYs break down across cause levels.",
        "how": "Box size reflects value; hierarchy shows parent-child structure.",
        "why": "Treemaps highlight composition and hierarchy at once.",
    },
    "bigcat_donut": {
        "title": "Big categories donut",
        "objective": "Show top-level category shares.",
        "how": "Slices represent Level 1 categories.",
        "why": "Donuts simplify the composition view.",
    },
    "relationships_scatter": {
        "title": "Relationship scatter",
        "objective": "Compare suicide rates with a selected indicator.",
        "how": "Each point is a country; color shows region.",
        "why": "Scatter plots highlight linear and non-linear patterns.",
    },
    "relationships_heatmap": {
        "title": "Correlation heatmap",
        "objective": "Summarize correlations among indicators.",
        "how": "Cells show Pearson correlation coefficients.",
        "why": "Heatmaps allow fast comparison across many pairs.",
    },
    "segmentation_map": {
        "title": "Cluster map",
        "objective": "Show how clusters distribute geographically.",
        "how": "Color indicates cluster label.",
        "why": "Spatial context clarifies cluster patterns.",
    },
    "segmentation_sizes": {
        "title": "Cluster sizes",
        "objective": "Show how many countries fall in each cluster.",
        "how": "Bars show country counts per cluster.",
        "why": "Size balance indicates cluster stability.",
    },
    "segmentation_profile": {
        "title": "Cluster profiles",
        "objective": "Compare indicator patterns across clusters.",
        "how": "Z-scores show relative highs and lows.",
        "why": "Profiles explain what makes a cluster distinct.",
    },
    "segmentation_k": {
        "title": "K selection",
        "objective": "Show silhouette scores for different k values.",
        "how": "Higher silhouette suggests better separation.",
        "why": "Helps justify the chosen number of clusters.",
    },
    "outliers_scatter": {
        "title": "Outlier scatter",
        "objective": "Highlight countries with unusual combinations.",
        "how": "Size or color flags outlier status or score.",
        "why": "Outlier scans support data mining insights.",
    },
    "ml_results": {
        "title": "Holdout metrics",
        "objective": "Compare baseline model accuracy on a test split.",
        "how": "Lower MAE and higher R2 indicate better fit.",
        "why": "Holdout metrics show performance on unseen data.",
    },
    "ml_cv": {
        "title": "Cross-validation metrics",
        "objective": "Assess stability across folds.",
        "how": "Mean and std show average performance and variance.",
        "why": "Cross-validation reduces sensitivity to a single split.",
    },
    "ml_pred_actual": {
        "title": "Predicted vs actual",
        "objective": "Check calibration and error spread.",
        "how": "Points near the diagonal indicate accurate predictions.",
        "why": "A visual check complements numeric metrics.",
    },
    "ml_feature_importance": {
        "title": "Feature importance",
        "objective": "Identify which features drive the model most.",
        "how": "Bars show RandomForest importance scores.",
        "why": "Importance supports interpretation and discussion.",
    },
}

V2_PAGE_GUIDES = {
    "v2_overview": {
        "objective": "Summarize synthetic patterns and regional trends at a glance.",
        "how": "Choose year, sex, and region to update the map and trend view.",
        "notes": "All values are synthetic and used for advanced demo workflows.",
    },
    "v2_clusters": {
        "objective": "Group countries into profiles using 2023 synthetic indicators.",
        "how": "Use the map to see cluster geography and the centers table to interpret clusters.",
        "notes": "Clusters are unsupervised and descriptive only.",
    },
    "v2_trajectory": {
        "objective": "Cluster countries by long-run suicide-rate trajectories.",
        "how": "Use scatter plots to understand slope, volatility, and recent change.",
        "notes": "Trajectories are based on synthetic time series (2000-2023).",
    },
    "v2_dtw": {
        "objective": "Cluster trajectory shapes using DTW distance.",
        "how": "Use the prototype trend to see the typical shape of each cluster.",
        "notes": "DTW focuses on curve shape rather than point-by-point alignment.",
    },
    "v2_network": {
        "objective": "Show similarity networks between countries.",
        "how": "Nodes are countries, edges are high cosine similarity; larger nodes are more central.",
        "notes": "Communities are detected with greedy modularity on the synthetic profiles.",
    },
    "v2_linked": {
        "objective": "Explore linked views with brushing and small multiples.",
        "how": "Select points in the scatter to filter the map and table.",
        "notes": "Small multiples use regional aggregates for context.",
    },
    "v2_forecasts": {
        "objective": "Compare classical and DL regional forecasts.",
        "how": "Choose the model, then read actual vs forecast lines.",
        "notes": "Forecasts are synthetic and demonstrate methodology only.",
    },
    "v2_backtest": {
        "objective": "Validate forecasting performance with a rolling backtest.",
        "how": "Compare actual vs predicted by region and review metrics.",
        "notes": "Backtest uses lag features and synthetic data only.",
    },
    "v2_scenario": {
        "objective": "Run what-if scenarios on a synthetic regression model.",
        "how": "Adjust inputs and watch the predicted suicide rate change.",
        "notes": "This is a toy model for demonstration, not causal inference.",
    },
    "v2_outliers": {
        "objective": "Highlight anomalous country profiles.",
        "how": "Outliers are flagged by IsolationForest on synthetic features.",
        "notes": "Outliers are statistical anomalies, not errors.",
    },
    "v2_patterns": {
        "objective": "Discover association rules among binned features.",
        "how": "Read rules by lift and interpret antecedents vs consequents.",
        "notes": "Rules are descriptive, not causal.",
    },
    "v2_methods": {
        "objective": "Document synthetic data generation, validation, and quality checks.",
        "how": "Review notes, dictionaries, validity reports, and GE outputs.",
        "notes": "All v2 outputs are synthetic and used for advanced demos only.",
    },
}

V2_CHART_GUIDES = {
    "v2_overview_kpis": {
        "title": "Overview KPIs",
        "objective": "Provide quick scale and baseline context for the selected year and sex.",
        "how": "KPIs summarize countries, average suicide rate, and average risk index.",
        "why": "KPIs set context before deeper exploration.",
    },
    "v2_overview_map": {
        "title": "Synthetic suicide-rate map",
        "objective": "Show geographic variation in suicide rates for the selected year and sex.",
        "how": "Darker color indicates higher synthetic rates; hover for details.",
        "why": "Maps expose spatial patterns quickly.",
    },
    "v2_overview_trend": {
        "title": "Regional trend",
        "objective": "Track the regional aggregate over time and show within-region spread.",
        "how": "Line is the aggregate; shaded band is the IQR across countries.",
        "why": "Combines central tendency with dispersion.",
    },
    "v2_overview_benchmark": {
        "title": "KPI benchmarking table",
        "objective": "Compare countries against global percentiles for the chosen metric.",
        "how": "Red is above p90; blue is below p10; sort by value.",
        "why": "Highlights extremes in a standardized way.",
    },
    "v2_clusters_map": {
        "title": "Profile cluster map",
        "objective": "Show where synthetic clusters appear geographically.",
        "how": "Each color is a cluster label.",
        "why": "Spatial context helps interpret cluster profiles.",
    },
    "v2_clusters_centers": {
        "title": "Cluster centers",
        "objective": "Summarize average indicators per cluster.",
        "how": "Each row is a cluster; columns are mean feature values.",
        "why": "Centers explain what makes clusters distinct.",
    },
    "v2_clusters_k": {
        "title": "K selection",
        "objective": "Document silhouette scores for different k values.",
        "how": "Higher silhouette indicates better separation.",
        "why": "Justifies the chosen number of clusters.",
    },
    "v2_traj_map": {
        "title": "Trajectory cluster map",
        "objective": "Show spatial distribution of trajectory clusters.",
        "how": "Each country is colored by trajectory cluster.",
        "why": "Highlights geographic differences in long-run trends.",
    },
    "v2_traj_scatter_slope": {
        "title": "Slope vs volatility",
        "objective": "Compare long-run direction and instability.",
        "how": "Right = increasing trends; up = more volatile.",
        "why": "Separates steady growth from noisy patterns.",
    },
    "v2_traj_scatter_mean": {
        "title": "Mean rate vs last-5y change",
        "objective": "Compare baseline levels with recent momentum.",
        "how": "Right = higher mean; up = recent increase.",
        "why": "Shows whether high levels are rising or stabilizing.",
    },
    "v2_traj_centers": {
        "title": "Trajectory centers",
        "objective": "Summarize cluster feature averages.",
        "how": "Each row is a cluster; columns are trajectory features.",
        "why": "Supports narrative for each cluster type.",
    },
    "v2_traj_k": {
        "title": "K selection",
        "objective": "Show silhouette metrics for trajectory clustering.",
        "how": "Higher silhouette indicates stronger separation.",
        "why": "Provides clustering justification.",
    },
    "v2_dtw_map": {
        "title": "DTW cluster map",
        "objective": "Map clusters based on time-series shape.",
        "how": "Colors represent DTW-based cluster labels.",
        "why": "DTW captures similar shapes even with timing shifts.",
    },
    "v2_dtw_prototype": {
        "title": "DTW prototype trend",
        "objective": "Show the typical trajectory shape for a cluster.",
        "how": "Line is the cluster center time series.",
        "why": "Helps interpret the meaning of each DTW cluster.",
    },
    "v2_dtw_k": {
        "title": "DTW K selection",
        "objective": "Show inertia and silhouette for DTW clustering.",
        "how": "Use these metrics to balance fit and parsimony.",
        "why": "Documents the chosen k.",
    },
    "v2_network_plot": {
        "title": "Country similarity network",
        "objective": "Visualize similarity structure among countries.",
        "how": "Nodes are countries; edges show similarity; size reflects centrality.",
        "why": "Networks reveal communities and hubs.",
    },
    "v2_network_sizes": {
        "title": "Community sizes",
        "objective": "Show how many countries belong to each community.",
        "how": "Rows list clusters and counts.",
        "why": "Size balance indicates community structure.",
    },
    "v2_network_central": {
        "title": "Top central countries",
        "objective": "Identify hubs with high betweenness or degree.",
        "how": "Higher centrality means more connections or bridging.",
        "why": "Highlights influential nodes in the network.",
    },
    "v2_network_edges": {
        "title": "Strongest edges",
        "objective": "List the most similar country pairs.",
        "how": "Higher similarity means closer profiles.",
        "why": "Shows the strongest ties in the network.",
    },
    "v2_linked_scatter": {
        "title": "Linked scatter",
        "objective": "Enable brushing to filter the rest of the view.",
        "how": "Select points; the map and table update.",
        "why": "Linked views support interactive exploration.",
    },
    "v2_linked_map": {
        "title": "Filtered map",
        "objective": "Show selected countries in geographic context.",
        "how": "Color encodes the chosen y metric.",
        "why": "Maps help validate spatial patterns in the selection.",
    },
    "v2_linked_table": {
        "title": "Filtered table",
        "objective": "Inspect the selected countries and values.",
        "how": "Table is sorted by the y-axis metric.",
        "why": "Tabular view supports detailed inspection.",
    },
    "v2_linked_multiples": {
        "title": "Regional small multiples",
        "objective": "Compare regional trends side by side.",
        "how": "Each panel is a region; y-axis is the selected metric.",
        "why": "Small multiples make comparisons easier.",
    },
    "v2_forecast_line": {
        "title": "Forecast line",
        "objective": "Compare actual vs forecast by region.",
        "how": "Different colors denote actual and forecast segments.",
        "why": "Shows model projections alongside history.",
    },
    "v2_forecast_metrics": {
        "title": "DL forecast metrics",
        "objective": "Summarize model error for DL forecasts.",
        "how": "Lower MAE/RMSE indicates better fit.",
        "why": "Quantifies forecast performance.",
    },
    "v2_backtest_line": {
        "title": "Backtest line",
        "objective": "Compare predicted vs actual in a rolling setup.",
        "how": "Closer alignment indicates better backtest fit.",
        "why": "Tests stability across time windows.",
    },
    "v2_backtest_metrics": {
        "title": "Backtest metrics",
        "objective": "Summarize backtest error by region.",
        "how": "Lower error means better generalization.",
        "why": "Supports model validation.",
    },
    "v2_scenario_pred": {
        "title": "Scenario prediction",
        "objective": "Show predicted suicide rate under current inputs.",
        "how": "Prediction updates instantly as sliders change.",
        "why": "Supports what-if analysis.",
    },
    "v2_sensitivity": {
        "title": "Sensitivity chart",
        "objective": "Show the effect of a 10% feature change.",
        "how": "Bars represent percent change in prediction.",
        "why": "Highlights local influence of each feature.",
    },
    "v2_quantile": {
        "title": "Quantile prediction bands",
        "objective": "Show uncertainty bands around predictions.",
        "how": "Band is q10-q90; q50 is the median prediction.",
        "why": "Communicates uncertainty, not just point estimates.",
    },
    "v2_quantile_metrics": {
        "title": "Quantile metrics",
        "objective": "Summarize quantile model accuracy.",
        "how": "Use these as diagnostic indicators.",
        "why": "Validates the interval model.",
    },
    "v2_explain_perm": {
        "title": "Permutation importance",
        "objective": "Explain which features matter most to the model.",
        "how": "Larger values mean bigger performance drop when shuffled.",
        "why": "Supports model interpretability.",
    },
    "v2_explain_pdp": {
        "title": "Partial dependence",
        "objective": "Show how predictions change across feature values.",
        "how": "Lines show marginal effect per feature.",
        "why": "Clarifies non-linear response patterns.",
    },
    "v2_outliers_scatter": {
        "title": "Outlier scatter",
        "objective": "Highlight anomalous countries in feature space.",
        "how": "Outliers are flagged and sized by score.",
        "why": "Supports anomaly detection insights.",
    },
    "v2_outliers_table": {
        "title": "Top anomalies",
        "objective": "List the highest-scoring outliers.",
        "how": "Rows show countries and their anomaly reasons.",
        "why": "Provides a concrete review list.",
    },
    "v2_patterns_table": {
        "title": "Association rules",
        "objective": "Show the strongest co-occurring patterns.",
        "how": "Sort by lift and review antecedents to consequents.",
        "why": "Reveals co-occurrence structures.",
    },
    "v2_patterns_interp": {
        "title": "Rule interpretation",
        "objective": "Explain the top rule in words.",
        "how": "Read antecedents -> consequents with lift and confidence.",
        "why": "Improves interpretability for non-technical readers.",
    },
}


@st.cache_data(hash_funcs={Path: lambda p: (str(p), p.stat().st_mtime) if p.exists() else str(p)})
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_exists(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing {label}: {path}")
        st.stop()
    return load_csv(path)


def load_report_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_csv(path)


@st.cache_data(hash_funcs={Path: lambda p: (str(p), p.stat().st_mtime) if p.exists() else str(p)})
def png_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_markdown_file(path: Path, fallback: str, strip_title: bool = False) -> None:
    if not path.exists():
        st.info(fallback)
        return
    content = path.read_text(encoding="utf-8")
    if strip_title:
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            content = "\n".join(lines[1:]).lstrip()
    st.markdown(content)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ml_baseline_features() -> pd.DataFrame:
    features_path = DATA_CLEAN / "ml_baseline_features.csv"
    if features_path.exists():
        df = load_csv(features_path)
        return numeric(
            df,
            [
                "age_standardized_suicide_rate_2021",
                "gbd_depression_dalys_rate_both",
                "gbd_addiction_death_rate_both",
                "gbd_selfharm_death_rate_both",
            ],
        )

    df = ensure_exists(DATA_CLEAN / "merged_ml_country.csv", "Merged ML dataset")
    df = numeric(
        df,
        [
            "age_standardized_suicide_rate_2021",
            "crude_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
            "gbd_addiction_death_rate_both",
            "gbd_selfharm_death_rate_male",
            "gbd_selfharm_death_rate_female",
            "gbd_selfharm_death_rate_both",
        ],
    )

    if "gbd_selfharm_death_rate_both" not in df.columns and {
        "gbd_selfharm_death_rate_male",
        "gbd_selfharm_death_rate_female",
    }.issubset(df.columns):
        df["gbd_selfharm_death_rate_both"] = df[
            ["gbd_selfharm_death_rate_male", "gbd_selfharm_death_rate_female"]
        ].mean(axis=1)

    if "region_name" in df.columns:
        df["region_name"] = df["region_name"].fillna("Unknown")
    else:
        df["region_name"] = "Unknown"

    if "income_group" in df.columns:
        df["income_group"] = df["income_group"].fillna("Unknown")
    else:
        df["income_group"] = "Unknown"

    if "data_quality" in df.columns:
        df["data_quality"] = df["data_quality"].fillna("Unknown")
    else:
        df["data_quality"] = "Unknown"

    group_cols = ["iso3", "location_name", "region_name", "income_group", "data_quality"]
    agg_cols = {
        "age_standardized_suicide_rate_2021": "mean",
        "gbd_depression_dalys_rate_both": "mean",
        "gbd_addiction_death_rate_both": "mean",
        "gbd_selfharm_death_rate_both": "mean",
    }
    if "crude_suicide_rate_2021" in df.columns:
        agg_cols["crude_suicide_rate_2021"] = "mean"

    df = df[group_cols + list(agg_cols.keys())].groupby(group_cols, as_index=False).agg(agg_cols)
    return df


def render_kpis(items: list[tuple[str, str]]) -> None:
    kpi_html = '<div class="kpi-grid">'
    for label, value in items:
        kpi_html += (
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>'
        )
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)


def render_page_guide_from_map(page_key: str, guide_map: dict[str, dict[str, str]]) -> None:
    guide = guide_map.get(page_key)
    if not guide:
        return
    st.markdown(
        f"""
<div class="guide-card">
  <div class="guide-title">Page guide</div>
  <div class="guide-line"><strong>Objective:</strong> {guide["objective"]}</div>
  <div class="guide-line"><strong>How to use:</strong> {guide["how"]}</div>
  <div class="guide-line"><strong>Notes:</strong> {guide["notes"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_page_guide(page_key: str) -> None:
    render_page_guide_from_map(page_key, V1_PAGE_GUIDES)


def render_chart_guide_from_map(
    chart_key: str,
    guide_map: dict[str, dict[str, str]],
    summary: str | None = None,
    key_prefix: str = "",
) -> None:
    guide = guide_map.get(chart_key)
    if not guide:
        return
    snapshot = summary or "Use the filters to update this view."
    st.markdown(
        f"<div class='chart-guide'><strong>Objective:</strong> {guide['objective']}<br/>"
        f"<strong>Snapshot:</strong> {snapshot}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Explain chart", key=f"{key_prefix}explain_{chart_key}"):
        def _dialog_body() -> None:
            st.markdown(f"**Objective:** {guide['objective']}")
            st.markdown(f"**How to read:** {guide['how']}")
            st.markdown(f"**Why this chart:** {guide['why']}")
            st.markdown(f"**Current snapshot:** {snapshot}")

        if hasattr(st, "dialog"):
            dialog = st.dialog(guide["title"])(_dialog_body)
            dialog()
        else:
            with st.expander(guide["title"], expanded=True):
                _dialog_body()


def render_chart_guide(chart_key: str, summary: str | None = None) -> None:
    render_chart_guide_from_map(chart_key, V1_CHART_GUIDES, summary=summary)


def fmt_value(val: float | int | None) -> str:
    if val is None or pd.isna(val):
        return "n/a"
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:,.2f}"


def summarize_top(df: pd.DataFrame, value_col: str, label_col: str = "location_name") -> str:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return "No data after filters."
    values = df[value_col].dropna()
    if values.empty:
        return "No data after filters."
    row = df.loc[values.idxmax()]
    return f"Highest: {row[label_col]} ({fmt_value(row[value_col])})."


def summarize_range(df: pd.DataFrame, value_col: str, label_col: str = "location_name") -> str:
    if df.empty or value_col not in df.columns:
        return "No data after filters."
    values = df[value_col].dropna()
    if values.empty:
        return "No data after filters."
    min_row = df.loc[values.idxmin()]
    max_row = df.loc[values.idxmax()]
    min_label = min_row[label_col] if label_col in df.columns else "min"
    max_label = max_row[label_col] if label_col in df.columns else "max"
    return (
        f"Range: {fmt_value(min_row[value_col])} ({min_label}) to "
        f"{fmt_value(max_row[value_col])} ({max_label})."
    )


def summarize_change(df: pd.DataFrame, value_col: str, year_col: str = "year") -> str:
    if df.empty or value_col not in df.columns:
        return "No data after filters."
    df = df.sort_values(year_col)
    if len(df) < 2:
        return "Not enough time points to show a trend."
    start = df.iloc[0][value_col]
    end = df.iloc[-1][value_col]
    if pd.isna(start) or pd.isna(end):
        return "Not enough data to compute change."
    delta = end - start
    pct = (delta / start * 100.0) if start not in (0, 0.0) else None
    pct_text = f" ({pct:+.1f}%)" if pct is not None else ""
    return f"Change: {fmt_value(start)} to {fmt_value(end)}{pct_text}."


def summarize_corr(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return "Not enough data to compute correlation."
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    if pd.isna(corr):
        return "Not enough data to compute correlation."
    return f"Correlation r={corr:.2f}."


def summarize_top_share(df: pd.DataFrame, value_col: str = "val", label_col: str = "cause_name") -> str:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return "No data after filters."
    values = df[value_col].dropna()
    if values.empty:
        return "No data after filters."
    total = values.sum()
    row = df.loc[values.idxmax()]
    share = (row[value_col] / total * 100.0) if total else None
    share_text = f" ({share:.1f}% of total)" if share is not None else ""
    return f"Largest category: {row[label_col]} ({fmt_value(row[value_col])}){share_text}."


def summarize_cluster_counts(counts: pd.DataFrame) -> str:
    if counts.empty or "cluster_label" not in counts.columns or "count" not in counts.columns:
        return "No clusters available."
    top = counts.iloc[0]
    return f"Largest cluster: {top['cluster_label']} ({int(top['count'])} countries)."


def summarize_outlier_counts(total: int, outliers: int) -> str:
    if total == 0:
        return "No data after filters."
    return f"Outliers flagged: {outliers} of {total} countries."


def page_v0_gallery() -> None:
    st.markdown(
        """
<style>
.v0-hero {
  background: linear-gradient(135deg, rgba(33, 94, 122, 0.12), rgba(242, 185, 80, 0.15));
  border: 1px solid rgba(33, 94, 122, 0.12);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 18px;
}
.v0-hero h1 {
  font-family: "Source Serif 4", serif;
  font-size: 30px;
  margin: 0 0 6px 0;
}
.v0-hero p { color: var(--muted); margin: 0; }
.v0-section {
  font-size: 18px;
  font-weight: 600;
  margin: 24px 0 12px 0;
  color: var(--ink);
}
.v0-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 12px 30px rgba(28, 27, 26, 0.08);
  margin-bottom: 16px;
}
.v0-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 6px;
}
.v0-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.v0-tag {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(31, 111, 139, 0.12);
  color: var(--accent);
}
.v0-image {
  width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.04);
  margin-bottom: 10px;
}
.v0-meta {
  display: grid;
  gap: 6px;
  font-size: 13px;
  color: var(--muted);
}
.v0-takeaway {
  margin: 10px 0;
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(242, 185, 80, 0.18);
  color: #6b4e00;
  font-size: 13px;
  font-weight: 600;
}
.v0-label { font-weight: 600; color: var(--ink); }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="v0-hero">
  <h1>v0 Static Visuals Gallery</h1>
  <p>Curated, presentation-ready visuals with clear objectives and chart rationale.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    version_dir = DATA_CLEAN.parent
    assets_dir = version_dir / "assets"
    manifest_path = assets_dir / "manifest.csv"
    if not manifest_path.exists():
        st.info("Missing v0/assets/manifest.csv. Generate it to power the gallery.")
        return

    manifest = load_csv(manifest_path)
    required_cols = {
        "id",
        "title",
        "dataset",
        "chart_type",
        "objective",
        "result",
        "why",
        "key_takeaway",
        "png",
        "html",
        "order",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        st.error(f"Manifest missing columns: {', '.join(sorted(missing))}")
        return

    manifest["order"] = pd.to_numeric(manifest["order"], errors="coerce").fillna(9999)
    datasets = sorted(manifest["dataset"].unique())
    chart_types = sorted(manifest["chart_type"].unique())

    filters = st.columns([2, 2, 1, 1])
    with filters[0]:
        dataset_sel = st.multiselect("Datasets", datasets, default=datasets)
    with filters[1]:
        chart_sel = st.multiselect("Chart types", chart_types, default=chart_types)
    with filters[2]:
        show_html = st.checkbox("Show HTML", value=False)
    with filters[3]:
        print_mode = st.checkbox("Print / PDF mode", value=False)

    if print_mode:
        st.markdown(
            """
<style>
section[data-testid="stSidebar"] { display: none; }
header, footer, div[data-testid="stToolbar"] { display: none; }
.stApp { background: #ffffff !important; }
.v0-card { box-shadow: none; border-color: #d9d2c8; }
.v0-hero { background: #ffffff; border-color: #d9d2c8; }
@media print {
  header, footer, section[data-testid="stSidebar"] { display: none !important; }
  .stApp { background: #ffffff !important; }
  .v0-card { break-inside: avoid; }
  button { display: none !important; }
}
</style>
""",
            unsafe_allow_html=True,
        )
        st.info("Print mode enabled. Use browser Print → Save as PDF for submission.")

    filtered = manifest[
        manifest["dataset"].isin(dataset_sel)
        & manifest["chart_type"].isin(chart_sel)
    ].copy()
    if filtered.empty:
        st.info("No charts match the current filters.")
        return

    dataset_blurbs = {
        "WHO Global": "WHO suicide indicators for 2021 with a focus on spatial patterns and rate comparability.",
        "WHO Regions": "Regional distribution views to compare spread and variability by sex.",
        "GBD Depression DALYs": "GBD DALYs burden for depressive disorders with age group rankings.",
        "GBD Mental & Substance Deaths": "Cause level deaths across mental and substance categories for cross cause comparison.",
        "GBD Age-Standardized Death Rate": "Age standardized death rates across causes and sexes for matrix scanning.",
        "GBD All-Cause Trends": "Longitudinal all cause trends to show metric trajectories over time.",
        "GBD Probability of Death": "Probability of death indicators highlighting spatial risk and top countries.",
        "GBD Risk Factors": "Risk factor composition view to show dominant contributors.",
        "GBD Anemia (YLDs)": "Anemia YLD rates across years by sex for temporal comparison.",
    }

    for dataset in datasets:
        if dataset not in dataset_sel:
            continue
        section = filtered[filtered["dataset"] == dataset].sort_values("order")
        if section.empty:
            continue
        st.markdown(f'<div class="v0-section">{html_lib.escape(dataset)}</div>', unsafe_allow_html=True)
        blurb = dataset_blurbs.get(dataset)
        if blurb:
            st.markdown(f'<div class="section-subtitle">{html_lib.escape(blurb)}</div>', unsafe_allow_html=True)

        rows = [section.iloc[i : i + 2] for i in range(0, len(section), 2)]
        for row_chunk in rows:
            cols = st.columns(len(row_chunk))
            for col, (_, row) in zip(cols, row_chunk.iterrows()):
                with col:
                    row_data = row.to_dict()
                    img_path = assets_dir / str(row["png"])
                    data_uri = png_to_data_uri(img_path)
                    title = html_lib.escape(str(row["title"]))
                    chart_type = html_lib.escape(str(row["chart_type"]))
                    objective = html_lib.escape(str(row["objective"]))
                    result = html_lib.escape(str(row["result"]))
                    why = html_lib.escape(str(row["why"]))
                    takeaway = html_lib.escape(str(row["key_takeaway"]))
                    img_html = (
                        f'<img class="v0-image" src="{data_uri}" alt="{title}">'
                        if data_uri
                        else '<div class="v0-image">Missing image</div>'
                    )
                    card = f"""
<div class="v0-card">
  <div class="v0-title">{title}</div>
  <div class="v0-tags">
    <span class="v0-tag">{chart_type}</span>
  </div>
  {img_html}
  <div class="v0-takeaway">Key takeaway: {takeaway}</div>
  <div class="v0-meta">
    <div><span class="v0-label">Objective:</span> {objective}</div>
    <div><span class="v0-label">Result:</span> {result}</div>
    <div><span class="v0-label">Why this chart:</span> {why}</div>
  </div>
</div>
"""
                    st.markdown(card, unsafe_allow_html=True)

                    dialog_key = f"v0_dialog_{row_data.get('id', '')}"
                    if st.button("Explain chart", key=dialog_key):
                        dialog_title = f"{row_data.get('title', 'Chart')} — Explanation"

                        @st.dialog(dialog_title)
                        def _chart_dialog() -> None:
                            st.markdown(f"**Dataset:** {row_data.get('dataset', '')}")
                            st.markdown(f"**Chart type:** {row_data.get('chart_type', '')}")
                            st.markdown(f"**Objective:** {row_data.get('objective', '')}")
                            st.markdown(f"**Result:** {row_data.get('result', '')}")
                            st.markdown(f"**Why this chart:** {row_data.get('why', '')}")
                            st.markdown(f"**Key takeaway:** {row_data.get('key_takeaway', '')}")

                        _chart_dialog()

                    if show_html:
                        html_path = assets_dir / str(row["html"])
                        if html_path.exists():
                            with st.expander("Interactive version", expanded=False):
                                st.components.v1.html(
                                    html_path.read_text(encoding="utf-8"),
                                    height=520,
                                    scrolling=True,
                                )


def load_v3_features(source: str) -> pd.DataFrame | None:
    path = DATA_CLEAN / f"v3_features_{source}.csv"
    if not path.exists():
        return None
    df = load_csv(path)
    df = numeric(df, ["suicide_rate"] + V3_NUMERIC_COLS)
    df = df.dropna(subset=["suicide_rate"] + V3_NUMERIC_COLS)
    return df


def train_v3_model(
    df: pd.DataFrame,
    cutoff: float,
) -> tuple[Pipeline, Pipeline | None, dict[str, float], list[str], list[str]]:
    if ColumnTransformer is None or LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for v3.")

    cutoff = float(cutoff)
    threshold = float(df["suicide_rate"].quantile(cutoff))
    y = (df["suicide_rate"] >= threshold).astype(int)

    cat_cols = [col for col in ["region_name", "income_group", "sex_name"] if col in df.columns]
    num_cols = [col for col in V3_NUMERIC_COLS if col in df.columns]

    X = df[cat_cols + num_cols]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    base = Pipeline([("prep", preprocessor), ("model", model)])
    base.fit(X, y)

    calibrated: Pipeline | None = None
    if CalibratedClassifierCV is not None and y.nunique() > 1 and len(df) >= 60:
        calib_base = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            calibrated = CalibratedClassifierCV(
                estimator=calib_base,
                method="isotonic",
                cv=3,
            )
        except TypeError:
            calibrated = CalibratedClassifierCV(
                base_estimator=calib_base,
                method="isotonic",
                cv=3,
            )
        calibrated.fit(X, y)

    metrics: dict[str, float] = {
        "rows": float(len(df)),
        "positive_rate": float(y.mean()),
        "threshold_rate": threshold,
    }
    if accuracy_score is not None and roc_auc_score is not None:
        predictor = calibrated if calibrated is not None else base
        proba = predictor.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics["train_accuracy"] = float(accuracy_score(y, preds))
        if y.nunique() > 1:
            metrics["train_auc"] = float(roc_auc_score(y, proba))
        if brier_score_loss is not None:
            metrics["train_brier"] = float(brier_score_loss(y, proba))

    return base, calibrated, metrics, cat_cols, num_cols


def v3_neighbors(
    df: pd.DataFrame,
    input_row: dict[str, object],
    threshold: float,
    k: int = 5,
) -> pd.DataFrame:
    pool = df.copy()
    if "sex_name" in df.columns and input_row.get("sex_name") in df["sex_name"].unique():
        pool = pool[pool["sex_name"] == input_row.get("sex_name")].copy()
        if len(pool) <= k:
            pool = df.copy()

    features = pool[V3_NUMERIC_COLS].copy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(features)
    input_vec = scaler.transform(pd.DataFrame([input_row])[V3_NUMERIC_COLS])
    dists = np.linalg.norm(matrix - input_vec, axis=1)
    pool = pool.assign(distance=dists)
    pool["high_risk"] = pool["suicide_rate"] >= threshold
    if "iso3" in pool.columns and input_row.get("iso3"):
        pool = pool[pool["iso3"] != input_row.get("iso3")]
    return pool.sort_values("distance").head(k)


def v3_reliability_curve(proba: np.ndarray, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"proba": proba, "y": y.astype(int)})
    df["bin"] = pd.qcut(df["proba"], q=bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_pred=("proba", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size"),
    )
    grouped = grouped.reset_index(drop=True)
    return grouped


def v3_feature_contributions(
    model: Pipeline,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    clf = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    x_trans = prep.transform(input_df)
    if hasattr(x_trans, "toarray"):
        x_vec = x_trans.toarray()[0]
    else:
        x_vec = np.asarray(x_trans)[0]
    contrib = x_vec * clf.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "contribution": contrib})

    def tidy_feature(name: str) -> str:
        name = name.replace("cat__", "").replace("num__", "")
        for prefix in ("region_name_", "income_group_", "sex_name_"):
            if name.startswith(prefix):
                key = prefix.replace("_", "").replace("name", "")
                return f"{key}={name[len(prefix):]}"
        return name.replace("_", " ")

    df["feature"] = df["feature"].map(tidy_feature)
    df["abs_contrib"] = df["contribution"].abs()
    return df.sort_values("abs_contrib", ascending=False)


def page_overview() -> None:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Story snapshot across WHO and GBD features.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("overview")

    who = ensure_exists(DATA_CLEAN / "who_2021_clean.csv", "WHO cleaned file")
    merged = ensure_exists(DATA_CLEAN / "merged_ml_country.csv", "Merged ML dataset")

    who = numeric(
        who,
        [
            "number_suicides_2021",
            "crude_suicide_rate_2021",
            "age_standardized_suicide_rate_2021",
        ],
    )
    who_both = who[who["sex_name"] == "Both sexes"].copy()
    who_both = who_both[who_both["iso3"].notna() & (who_both["iso3"].astype(str) != "")]

    kpis = [
        ("Countries", f"{who_both['iso3'].nunique():,}"),
        ("Avg age-std rate", f"{who_both['age_standardized_suicide_rate_2021'].mean():.2f}"),
        ("Median age-std rate", f"{who_both['age_standardized_suicide_rate_2021'].median():.2f}"),
    ]
    render_kpis(kpis)
    kpi_summary = (
        f"{who_both['iso3'].nunique():,} countries; "
        f"mean {fmt_value(who_both['age_standardized_suicide_rate_2021'].mean())}, "
        f"median {fmt_value(who_both['age_standardized_suicide_rate_2021'].median())}."
    )
    render_chart_guide("overview_kpis", summary=kpi_summary)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        top10 = (
            who_both.nlargest(10, "age_standardized_suicide_rate_2021")
            .sort_values("age_standardized_suicide_rate_2021")
        )
        fig = px.bar(
            top10,
            x="age_standardized_suicide_rate_2021",
            y="location_name",
            orientation="h",
            color="age_standardized_suicide_rate_2021",
            color_continuous_scale="Cividis",
            labels={"age_standardized_suicide_rate_2021": "Age-std suicide rate"},
            title="Top 10 suicide rates (Both sexes)",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide(
            "overview_top10",
            summary=summarize_top(top10, "age_standardized_suicide_rate_2021"),
        )

    with col2:
        fig = px.box(
            who_both,
            x="region_name",
            y="age_standardized_suicide_rate_2021",
            color="region_name",
            title="Regional spread (age-standardized rate)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        region_med = (
            who_both.groupby("region_name", as_index=False)["age_standardized_suicide_rate_2021"]
            .median()
            .sort_values("age_standardized_suicide_rate_2021")
        )
        if region_med.empty:
            region_summary = "No data after filters."
        else:
            low = region_med.iloc[0]
            high = region_med.iloc[-1]
            region_summary = (
                f"Highest median: {high['region_name']} "
                f"({fmt_value(high['age_standardized_suicide_rate_2021'])}); "
                f"lowest median: {low['region_name']} "
                f"({fmt_value(low['age_standardized_suicide_rate_2021'])})."
            )
        render_chart_guide("overview_region_box", summary=region_summary)

    st.markdown("### Model demo signal (quick check)")
    merged = numeric(
        merged,
        [
            "age_standardized_suicide_rate_2021",
            "gbd_depression_dalys_rate_both",
        ],
    )
    merged_25 = merged[merged["age_name"] == "25+ years"].copy()
    corr = merged_25[
        ["age_standardized_suicide_rate_2021", "gbd_depression_dalys_rate_both"]
    ].corr().iloc[0, 1]
    render_kpis([("Suicide vs Depression corr", f"{corr:.2f}")])
    render_chart_guide("overview_corr", summary=f"Correlation r={corr:.2f} (25+ years).")


def page_who_explorer() -> None:
    st.markdown('<div class="section-title">WHO Suicide Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Global and regional suicide patterns (2021).</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("who")

    who = ensure_exists(DATA_CLEAN / "who_2021_clean.csv", "WHO cleaned file")
    who = numeric(
        who,
        [
            "number_suicides_2021",
            "crude_suicide_rate_2021",
            "age_standardized_suicide_rate_2021",
        ],
    )

    sex = st.selectbox("Sex", sorted(who["sex_name"].unique()))
    metric = st.selectbox(
        "Metric", ["age_standardized_suicide_rate_2021", "crude_suicide_rate_2021"]
    )
    who_sel = who[who["sex_name"] == sex].copy()
    who_sel = who_sel[who_sel["iso3"].notna() & (who_sel["iso3"].astype(str) != "")]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.choropleth(
            who_sel,
            locations="iso3",
            color=metric,
            hover_name="location_name",
            color_continuous_scale="Cividis",
            title="Age-standardized suicide rate (map)" if metric.startswith("age") else "Crude rate (map)",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("who_map", summary=summarize_range(who_sel, metric))

    with col2:
        top = who_sel.nlargest(10, metric).sort_values(metric)
        bottom = who_sel.nsmallest(10, metric).sort_values(metric)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top[metric],
                y=top["location_name"],
                orientation="h",
                name="Top 10",
                marker_color="#1f6f8b",
            )
        )
        fig.add_trace(
            go.Bar(
                x=bottom[metric],
                y=bottom["location_name"],
                orientation="h",
                name="Bottom 10",
                marker_color="#f2b950",
            )
        )
        fig.update_layout(
            title="Top/Bottom countries",
            barmode="group",
            legend_title_text="",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        if top.empty or bottom.empty:
            rank_summary = "No data after filters."
        else:
            rank_summary = (
                f"Top: {top.iloc[-1]['location_name']} ({fmt_value(top.iloc[-1][metric])}); "
                f"Bottom: {bottom.iloc[0]['location_name']} ({fmt_value(bottom.iloc[0][metric])})."
            )
        render_chart_guide("who_top_bottom", summary=rank_summary)

    st.markdown("### Sex comparison")
    country = st.selectbox("Country", sorted(who["location_name"].unique()))
    compare = who[who["location_name"] == country].copy()
    fig = px.bar(
        compare,
        x="sex_name",
        y=metric,
        color="sex_name",
        title=f"{country} | {metric.replace('_', ' ')}",
    )
    st.plotly_chart(fig, use_container_width=True)
    if compare.empty:
        sex_summary = "No data after filters."
    else:
        max_row = compare.loc[compare[metric].idxmax()]
        sex_summary = f"Highest: {max_row['sex_name']} ({fmt_value(max_row[metric])})."
    render_chart_guide("who_sex_compare", summary=sex_summary)

    st.markdown("### Crude vs age-standardized")
    scatter = who_sel.dropna(subset=["crude_suicide_rate_2021", "age_standardized_suicide_rate_2021"])
    fig = px.scatter(
        scatter,
        x="crude_suicide_rate_2021",
        y="age_standardized_suicide_rate_2021",
        color="region_name",
        hover_name="location_name",
        title="Crude vs age-standardized rate",
    )
    st.plotly_chart(fig, use_container_width=True)
    render_chart_guide(
        "who_crude_vs_age",
        summary=summarize_corr(scatter, "crude_suicide_rate_2021", "age_standardized_suicide_rate_2021"),
    )


def page_depression() -> None:
    st.markdown('<div class="section-title">Depression Burden (DALYs)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">GBD DALYs rate for depressive disorders.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("depression")
    df = ensure_exists(DATA_CLEAN / "gbd_depression_dalys_clean.csv", "GBD depression clean file")
    df = numeric(df, ["val"])
    df = df[
        (df["cause_name"] == "Depressive disorders")
        & (df["measure_name"] == "DALYs (Disability-Adjusted Life Years)")
        & (df["metric_name"] == "Rate")
    ]
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]

    df_both_all = df[df["sex_name"] == "Both"].copy()
    age = st.selectbox("Age group", sorted(df_both_all["age_name"].unique()))
    df_both = df_both_all[df_both_all["age_name"] == age]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.choropleth(
            df_both,
            locations="iso3",
            color="val",
            hover_name="location_name",
            color_continuous_scale="Cividis",
            title=f"DALYs rate (Both sexes, {age})",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("depression_map", summary=summarize_range(df_both, "val"))
    with col2:
        top = df_both.nlargest(20, "val").sort_values("val")
        fig = px.bar(
            top,
            x="val",
            y="location_name",
            orientation="h",
            title="Top 20 countries",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("depression_top20", summary=summarize_top(top, "val"))

    st.markdown("### By age group")
    age_summary = (
        df_both_all.groupby("age_name", as_index=False)["val"]
        .mean()
    )
    age_order = ["<20 years", "20-24 years", "25+ years"]
    fig = px.bar(
        age_summary,
        x="age_name",
        y="val",
        title="Average DALYs rate by age group",
        category_orders={"age_name": age_order},
    )
    fig.update_yaxes(dtick=100)
    fig.update_layout(height=520, margin=dict(l=50, r=30, t=70, b=50))
    st.plotly_chart(fig, use_container_width=True)
    if age_summary.empty:
        age_summary_text = "No data after filters."
    else:
        top_age = age_summary.loc[age_summary["val"].idxmax()]
        age_summary_text = f"Highest average: {top_age['age_name']} ({fmt_value(top_age['val'])})."
    render_chart_guide("depression_age_bar", summary=age_summary_text)


def page_addiction() -> None:
    st.markdown('<div class="section-title">Addiction (Deaths Rate)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">GBD substance-use mortality rates.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("addiction")
    df = ensure_exists(DATA_CLEAN / "gbd_addiction_clean.csv", "GBD addiction clean file")
    df = numeric(df, ["val"])
    df = df[
        (df["measure_name"] == "Deaths")
        & (df["metric_name"] == "Rate")
        & (df["year"].astype(str) == "2023")
    ]
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]

    causes = sorted(df["cause_name"].unique())
    cause = st.selectbox("Cause", causes)
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    filtered = df[(df["cause_name"] == cause) & (df["sex_name"] == sex)]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.choropleth(
            filtered,
            locations="iso3",
            color="val",
            hover_name="location_name",
            color_continuous_scale="Cividis",
            title=f"{cause} deaths rate ({sex})",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("addiction_map", summary=summarize_range(filtered, "val"))
    with col2:
        top = filtered.nlargest(20, "val").sort_values("val")
        fig = px.bar(
            top,
            x="val",
            y="location_name",
            orientation="h",
            title="Top 20 countries",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("addiction_top20", summary=summarize_top(top, "val"))

    st.markdown("### Sex comparison")
    country = st.selectbox("Country", sorted(df["location_name"].unique()))
    compare = df[(df["location_name"] == country) & (df["cause_name"] == cause)].copy()
    compare = compare.groupby("sex_name", as_index=False)["val"].mean()
    fig = px.bar(
        compare,
        x="sex_name",
        y="val",
        color="sex_name",
        title=f"{country} | {cause} deaths rate",
    )
    st.plotly_chart(fig, use_container_width=True)
    if compare.empty:
        sex_summary = "No data after filters."
    else:
        max_row = compare.loc[compare["val"].idxmax()]
        sex_summary = f"Highest: {max_row['sex_name']} ({fmt_value(max_row['val'])})."
    render_chart_guide("addiction_sex_compare", summary=sex_summary)


def page_selfharm() -> None:
    st.markdown('<div class="section-title">Self-harm (Deaths Rate)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">GBD self-harm mortality patterns.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("selfharm")
    df = ensure_exists(DATA_CLEAN / "gbd_selfharm_clean.csv", "GBD self-harm clean file")
    df = numeric(df, ["val"])
    df = df[
        (df["measure_name"] == "Deaths")
        & (df["metric_name"] == "Rate")
        & (df["cause_name"] == "Self-harm")
    ]
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]

    age = st.selectbox("Age group", sorted(df["age_name"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    df = df[(df["age_name"] == age) & (df["sex_name"] == sex)]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.choropleth(
            df,
            locations="iso3",
            color="val",
            hover_name="location_name",
            color_continuous_scale="Cividis",
            title=f"Self-harm deaths rate ({sex}, {age})",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("selfharm_map", summary=summarize_range(df, "val"))
    with col2:
        top = df.nlargest(20, "val").sort_values("val")
        fig = px.bar(
            top,
            x="val",
            y="location_name",
            orientation="h",
            title="Top 20 countries",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("selfharm_top20", summary=summarize_top(top, "val"))

    st.markdown("### Sex comparison")
    base = ensure_exists(DATA_CLEAN / "gbd_selfharm_clean.csv", "GBD self-harm clean file")
    base = numeric(base, ["val"])
    base = base[
        (base["measure_name"] == "Deaths")
        & (base["metric_name"] == "Rate")
        & (base["cause_name"] == "Self-harm")
        & (base["age_name"] == age)
    ]
    country = st.selectbox("Country", sorted(base["location_name"].unique()))
    compare = base[base["location_name"] == country]
    fig = px.bar(
        compare,
        x="sex_name",
        y="val",
        color="sex_name",
        title=f"{country} | Self-harm rate ({age})",
    )
    st.plotly_chart(fig, use_container_width=True)
    if compare.empty:
        sex_summary = "No data after filters."
    else:
        max_row = compare.loc[compare["val"].idxmax()]
        sex_summary = f"Highest: {max_row['sex_name']} ({fmt_value(max_row['val'])})."
    render_chart_guide("selfharm_sex_compare", summary=sex_summary)

    if base["cause_name"].nunique() > 1:
        st.markdown("### Methods breakdown")
        methods = base.groupby("cause_name", as_index=False)["val"].mean()
        fig = px.bar(
            methods,
            x="cause_name",
            y="val",
            title="Self-harm methods (average rate)",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("selfharm_methods", summary=summarize_top(methods, "val", "cause_name"))
    else:
        st.info("Methods breakdown not available in the filtered self-harm dataset.")


def page_prob_death() -> None:
    st.markdown('<div class="section-title">Probability of Death</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Interpretation differs from rates: it is a probability, not a per-100k rate.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("prob_death")
    df = ensure_exists(DATA_CLEAN / "gbd_prob_death_clean.csv", "GBD probability-of-death file")
    df = numeric(df, ["val"])
    df = df[df["metric_name"] == "Probability of death"]
    df = df[df["iso3"].notna() & (df["iso3"].astype(str) != "")]

    cause = st.selectbox("Cause", sorted(df["cause_name"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    age = st.selectbox("Age group", sorted(df["age_name"].unique()))
    df = df[(df["cause_name"] == cause) & (df["sex_name"] == sex) & (df["age_name"] == age)]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.choropleth(
            df,
            locations="iso3",
            color="val",
            hover_name="location_name",
            color_continuous_scale="Cividis",
            title=f"Probability of death | {cause} ({sex}, {age})",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("probdeath_map", summary=summarize_range(df, "val"))
    with col2:
        rank = df.nlargest(20, "val").sort_values("val")
        fig = px.bar(
            rank,
            x="val",
            y="location_name",
            orientation="h",
            title="Top 20 countries",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("probdeath_top20", summary=summarize_top(rank, "val"))


def page_allcause_trends() -> None:
    st.markdown('<div class="section-title">All-cause Trends</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">DALYs trend across countries, WHO regions, and global aggregates.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("allcause")
    df = ensure_exists(CONTEXT_DIR / "context_allcauses_trend.csv", "Context all-cause trends")
    df = numeric(df, ["val"])

    location_type = st.selectbox("Location type", sorted(df["location_type"].unique()))
    subset = df[df["location_type"] == location_type]
    location = st.selectbox("Location", sorted(subset["location_name"].unique()))
    sex = st.selectbox("Sex", sorted(subset["sex_name"].unique()))
    age = st.selectbox("Age group", sorted(subset["age_name"].unique()))
    metric = st.selectbox("Metric", sorted(subset["metric_name"].unique()))

    filtered = subset[
        (subset["location_name"] == location)
        & (subset["sex_name"] == sex)
        & (subset["age_name"] == age)
        & (subset["metric_name"] == metric)
    ]
    filtered = filtered.sort_values("year")
    fig = px.line(
        filtered,
        x="year",
        y="val",
        markers=True,
        title=f"{location} | {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)
    render_chart_guide("allcause_trend", summary=summarize_change(filtered, "val"))
    st.dataframe(filtered.sort_values("year"), use_container_width=True)


def page_big_categories() -> None:
    st.markdown('<div class="section-title">Big Categories</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">GBD aggregate locations for big-category DALYs.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("big_categories")
    df = ensure_exists(CONTEXT_DIR / "context_big_categories_2023.csv", "Context big categories")
    df = numeric(df, ["val"])

    location = st.selectbox("Aggregate location", sorted(df["location_name"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    age = st.selectbox("Age group", sorted(df["age_name"].unique()))
    metric = st.selectbox("Metric", sorted(df["metric_name"].unique()))
    chart = st.selectbox("Chart type", ["Treemap", "Donut"])

    filtered = df[
        (df["location_name"] == location)
        & (df["sex_name"] == sex)
        & (df["age_name"] == age)
        & (df["metric_name"] == metric)
    ]
    filtered = filtered.groupby("cause_name", as_index=False)["val"].mean()
    summary_df = filtered[filtered["cause_name"] != "All causes"].copy()
    if summary_df.empty:
        summary_df = filtered.copy()
    summary_text = summarize_top_share(summary_df)

    chart_key = "bigcat_treemap" if chart == "Treemap" else "bigcat_donut"
    if chart == "Treemap":
        val_map = dict(zip(filtered["cause_name"], filtered["val"]))
        all_val = val_map.get("All causes")
        comm_val = val_map.get("Communicable, maternal, neonatal, and nutritional diseases")
        ncd_val = val_map.get("Non-communicable diseases")
        inj_val = val_map.get("Injuries")
        sub_val = val_map.get("Substance use disorders")
        alc_val = val_map.get("Alcohol use disorders")
        drug_val = val_map.get("Drug use disorders")

        rows = []

        def add_leaf(level1: str, level2: str | None, level3: str | None, value: float | None) -> None:
            if value is None or pd.isna(value) or value <= 0:
                return
            rows.append(
                {
                    "level0": "All causes",
                    "level1": level1,
                    "level2": level2 or "",
                    "level3": level3 or "",
                    "val": value,
                }
            )

        if comm_val is not None:
            add_leaf("Communicable, maternal, neonatal, and nutritional diseases", "", "", float(comm_val))
        if inj_val is not None:
            add_leaf("Injuries", "", "", float(inj_val))

        if ncd_val is not None:
            if sub_val is not None:
                if alc_val is not None:
                    add_leaf(
                        "Non-communicable diseases",
                        "Substance use disorders",
                        "Alcohol use disorders",
                        float(alc_val),
                    )
                if drug_val is not None:
                    add_leaf(
                        "Non-communicable diseases",
                        "Substance use disorders",
                        "Drug use disorders",
                        float(drug_val),
                    )
                other_sub = sub_val
                if alc_val is not None:
                    other_sub -= alc_val
                if drug_val is not None:
                    other_sub -= drug_val
                add_leaf(
                    "Non-communicable diseases",
                    "Substance use disorders",
                    "Other substance use disorders",
                    float(other_sub),
                )

                other_ncd = ncd_val - sub_val
                add_leaf(
                    "Non-communicable diseases",
                    "Other non-communicable diseases",
                    "",
                    float(other_ncd),
                )
            else:
                add_leaf("Non-communicable diseases", "", "", float(ncd_val))

        if all_val is not None and None not in (comm_val, ncd_val, inj_val):
            other_all = all_val - comm_val - ncd_val - inj_val
            add_leaf("Other causes", "", "", float(other_all))

        tree = pd.DataFrame(rows)
        if tree.empty:
            tree = filtered.rename(columns={"cause_name": "level1"})
            tree["level0"] = "All causes"
            tree["level2"] = ""
            tree["level3"] = ""

        fig = px.treemap(
            tree,
            path=["level0", "level1", "level2", "level3"],
            values="val",
            title=f"{location} | {metric}",
        )
    else:
        top_level = [
            "Communicable, maternal, neonatal, and nutritional diseases",
            "Non-communicable diseases",
            "Injuries",
        ]
        donut = filtered[filtered["cause_name"].isin(top_level)].copy()
        if donut.empty:
            donut = filtered.copy()
        fig = px.pie(
            donut,
            names="cause_name",
            values="val",
            hole=0.45,
            title=f"{location} | {metric}",
        )
    st.plotly_chart(fig, use_container_width=True)
    render_chart_guide(chart_key, summary=summary_text)


def page_segmentation() -> None:
    st.markdown('<div class="section-title">Country Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Unsupervised clustering of country profiles.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("segmentation")

    clusters_path = DATA_CLEAN / "segmentation_clusters.csv"
    if not clusters_path.exists():
        st.info("Run src/08_segmentation_outliers.py to generate segmentation outputs.")
        return

    df = load_csv(clusters_path)
    feature_labels = {
        "age_standardized_suicide_rate_2021": "Suicide rate (age-std)",
        "gbd_depression_dalys_rate_both": "Depression DALYs rate",
        "gbd_addiction_death_rate_both": "Addiction deaths rate",
        "gbd_selfharm_death_rate_both": "Self-harm deaths rate",
    }
    feature_cols = list(feature_labels.keys())
    df = numeric(df, feature_cols)

    k = df["cluster"].nunique() if "cluster" in df.columns else 0
    render_kpis(
        [
            ("Countries", f"{df['iso3'].nunique():,}"),
            ("Clusters (k)", f"{k}"),
            ("Outliers flagged", f"{df['is_outlier'].sum():,}" if "is_outlier" in df.columns else "n/a"),
        ]
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.choropleth(
            df,
            locations="iso3",
            color="cluster_label",
            hover_name="location_name",
            title="Cluster map",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        counts = df["cluster_label"].value_counts().reset_index()
        counts.columns = ["cluster_label", "count"]
        fig = px.bar(
            counts,
            x="cluster_label",
            y="count",
            title="Cluster sizes",
            color="cluster_label",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide("segmentation_sizes", summary=summarize_cluster_counts(counts))
    render_chart_guide(
        "segmentation_map",
        summary=summarize_cluster_counts(df["cluster_label"].value_counts().reset_index().rename(
            columns={"index": "cluster_label", "cluster_label": "count"}
        )),
    )

    centers = load_report_csv(REPORT_DIR / "segmentation_cluster_centers.csv")
    if centers is not None and not centers.empty:
        z_cols = [f"{col}_z" for col in feature_cols]
        if all(col in centers.columns for col in z_cols):
            profile = centers[["cluster_label"] + z_cols].melt(
                id_vars="cluster_label",
                value_vars=z_cols,
                var_name="feature",
                value_name="z_score",
            )
            profile["feature"] = profile["feature"].str.replace("_z", "", regex=False)
            profile["feature"] = profile["feature"].map(feature_labels)
            fig = px.line(
                profile,
                x="feature",
                y="z_score",
                color="cluster_label",
                markers=True,
                title="Cluster profile (z-score)",
            )
            st.plotly_chart(fig, use_container_width=True)
            render_chart_guide("segmentation_profile", summary="Z-scores compare each cluster to the overall mean.")

        st.markdown("### Cluster centers (original units)")
        center_cols = ["cluster_label", "count"] + feature_cols
        display_cols = [col for col in center_cols if col in centers.columns]
        st.dataframe(centers[display_cols], use_container_width=True)

    k_metrics = load_report_csv(REPORT_DIR / "segmentation_k_selection.csv")
    if k_metrics is not None and not k_metrics.empty:
        st.markdown("### K selection (silhouette)")
        st.dataframe(k_metrics, use_container_width=True)
        best_row = k_metrics.loc[k_metrics["silhouette"].idxmax()] if "silhouette" in k_metrics.columns else None
        if best_row is not None:
            k_summary = f"Best silhouette at k={int(best_row['k'])} ({best_row['silhouette']:.2f})."
        else:
            k_summary = "Silhouette scores summarize cluster separation."
        render_chart_guide("segmentation_k", summary=k_summary)


def page_outliers() -> None:
    st.markdown('<div class="section-title">Outliers & Alerts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Countries with unusual indicator patterns.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("outliers")

    features_path = DATA_CLEAN / "segmentation_features.csv"
    if not features_path.exists():
        st.info("Run src/08_segmentation_outliers.py to generate outlier outputs.")
        return

    df = load_csv(features_path)
    feature_labels = {
        "age_standardized_suicide_rate_2021": "Suicide rate (age-std)",
        "gbd_depression_dalys_rate_both": "Depression DALYs rate",
        "gbd_addiction_death_rate_both": "Addiction deaths rate",
        "gbd_selfharm_death_rate_both": "Self-harm deaths rate",
    }
    feature_cols = list(feature_labels.keys())
    df = numeric(df, feature_cols)

    x_col = st.selectbox("X axis", feature_cols, index=0)
    y_col = st.selectbox("Y axis", feature_cols, index=1)
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="is_outlier" if "is_outlier" in df.columns else None,
        size="outlier_score" if "outlier_score" in df.columns else None,
        hover_name="location_name",
        title=f"{feature_labels[x_col]} vs {feature_labels[y_col]}",
    )
    st.plotly_chart(fig, use_container_width=True)
    outlier_count = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0
    render_chart_guide(
        "outliers_scatter",
        summary=summarize_outlier_counts(len(df), outlier_count),
    )

    outliers = df[df["is_outlier"]] if "is_outlier" in df.columns else pd.DataFrame()
    if outliers.empty:
        st.info("No outliers detected with the current threshold.")
    else:
        cols = [
            "location_name",
            "region_name",
            "income_group",
            "outlier_score",
            "outlier_reason",
        ] + feature_cols
        cols = [col for col in cols if col in outliers.columns]
        st.markdown("### Top anomalies")
        st.dataframe(
            outliers.sort_values("outlier_score", ascending=False)[cols].head(20),
            use_container_width=True,
        )


def page_v2_overview() -> None:
    st.markdown('<div class="section-title">v2 Synthetic Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Synthetic long-panel data for advanced demos.</div>',
        unsafe_allow_html=True,
    )
    st.info("Synthetic data for demonstration only.")
    render_page_guide_from_map("v2_overview", V2_PAGE_GUIDES)

    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not data_path.exists():
        st.info("Run src/v2_generate_synth.py and src/v2_analytics.py for v2 outputs.")
        return

    df = load_csv(data_path)
    df = numeric(
        df,
        [
            "suicide_rate",
            "depression_dalys_rate",
            "addiction_death_rate",
            "selfharm_death_rate",
            "risk_index",
            "population",
        ],
    )

    year = st.selectbox("Year", sorted(df["year"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    subset = df[(df["year"] == year) & (df["sex_name"] == sex)]

    render_kpis(
        [
            ("Countries", f"{subset['iso3'].nunique():,}"),
            ("Avg suicide rate", f"{subset['suicide_rate'].mean():.2f}"),
            ("Avg risk index", f"{subset['risk_index'].mean():.2f}"),
        ]
    )
    kpi_summary = (
        f"{subset['iso3'].nunique():,} countries; "
        f"avg suicide rate {fmt_value(subset['suicide_rate'].mean())}; "
        f"avg risk index {fmt_value(subset['risk_index'].mean())}."
    )
    render_chart_guide_from_map("v2_overview_kpis", V2_CHART_GUIDES, summary=kpi_summary, key_prefix="v2_")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.choropleth(
            subset,
            locations="iso3",
            color="suicide_rate",
            hover_name="location_name",
            color_continuous_scale="Reds",
            title=f"Synthetic suicide rate ({year}, {sex})",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_overview_map",
            V2_CHART_GUIDES,
            summary=summarize_range(subset, "suicide_rate"),
            key_prefix="v2_",
        )

    with col2:
        region = st.selectbox("Region", sorted(df["region_name"].unique()))
        trend_view = st.selectbox(
            "Trend view",
            ["Region + IQR band", "Region aggregate", "Country trajectories"],
        )
        trend = df[(df["region_name"] == region) & (df["sex_name"] == sex)].copy()
        trend["year"] = pd.to_numeric(trend["year"], errors="coerce")
        trend = trend.dropna(subset=["year", "suicide_rate"]).sort_values("year")

        region_path = DATA_CLEAN / "synth_region_year.csv"
        region_df = None
        if region_path.exists():
            region_df = load_csv(region_path)
            region_df = numeric(region_df, ["suicide_rate"])
            region_df["year"] = pd.to_numeric(region_df["year"], errors="coerce")
            region_df = region_df[
                (region_df["region_name"] == region) & (region_df["sex_name"] == sex)
            ].dropna(subset=["year", "suicide_rate"]).sort_values("year")
            if region_df.empty:
                region_df = None

        trend_summary = None
        if trend_view == "Country trajectories":
            fig = px.line(
                trend,
                x="year",
                y="suicide_rate",
                line_group="location_name",
                title=f"{region} trend ({sex})",
            )
            fig.update_traces(line=dict(color="#1f6f8b"), opacity=0.35)
            fig.update_layout(showlegend=False)
            st.caption("Each line is a country. The region trend is the overall band of country trajectories.")
            if not trend.empty:
                trend_summary = summarize_change(
                    trend.groupby("year", as_index=False)["suicide_rate"].median(),
                    "suicide_rate",
                )
        elif trend_view == "Region aggregate":
            if region_df is None:
                st.info("Region aggregates not available. Showing country trajectories.")
                fig = px.line(
                    trend,
                    x="year",
                    y="suicide_rate",
                    line_group="location_name",
                    title=f"{region} trend ({sex})",
                )
                fig.update_traces(line=dict(color="#1f6f8b"), opacity=0.35)
                fig.update_layout(showlegend=False)
                if not trend.empty:
                    trend_summary = summarize_change(
                        trend.groupby("year", as_index=False)["suicide_rate"].median(),
                        "suicide_rate",
                    )
            else:
                fig = px.line(
                    region_df,
                    x="year",
                    y="suicide_rate",
                    title=f"{region} trend ({sex})",
                    markers=True,
                )
                st.caption("This line is the population-weighted regional aggregate.")
                trend_summary = summarize_change(region_df, "suicide_rate")
        else:
            fig = go.Figure()
            if not trend.empty:
                band = (
                    trend.groupby("year")["suicide_rate"]
                    .quantile([0.25, 0.75])
                    .unstack()
                    .reset_index()
                    .rename(columns={0.25: "p25", 0.75: "p75"})
                )
                fig.add_trace(
                    go.Scatter(
                        x=band["year"],
                        y=band["p75"],
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=band["year"],
                        y=band["p25"],
                        fill="tonexty",
                        fillcolor="rgba(31, 111, 139, 0.18)",
                        line=dict(width=0),
                        name="Country IQR",
                        hoverinfo="skip",
                    )
                )
            if region_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=region_df["year"],
                        y=region_df["suicide_rate"],
                        mode="lines+markers",
                        name="Region aggregate",
                        line=dict(color="#1f6f8b", width=3.5),
                    )
                )
            fig.update_layout(title=f"{region} trend ({sex})", yaxis_title="suicide_rate", xaxis_title="year")
            st.caption("Line = regional aggregate; shaded band = country interquartile range (25th–75th).")
            trend_summary = summarize_change(region_df, "suicide_rate") if region_df is not None else None
        changepoints_path = REPORT_DIR / "v2_changepoints.csv"
        if sex == "Both" and changepoints_path.exists():
            cp = load_csv(changepoints_path)
            if not cp.empty and {"region_name", "year", "suicide_rate"}.issubset(cp.columns):
                cp = cp[cp["region_name"] == region].copy()
                cp["year"] = pd.to_numeric(cp["year"], errors="coerce")
                cp = cp.dropna(subset=["year", "suicide_rate"])
                if not cp.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=cp["year"],
                            y=cp["suicide_rate"],
                            mode="markers",
                            name="Change-point",
                            marker=dict(color="#f08a5d", size=9, symbol="x"),
                        )
                    )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_overview_trend",
            V2_CHART_GUIDES,
            summary=trend_summary,
            key_prefix="v2_",
        )

    benchmarks_path = REPORT_DIR / "v2_kpi_benchmarks.csv"
    if benchmarks_path.exists():
        st.markdown("### KPI Benchmarking (2021 global percentiles)")
        benchmarks = load_csv(benchmarks_path)
        benchmarks = numeric(benchmarks, ["p10", "median", "p90"])
        feature_labels = {
            "suicide_rate": "Suicide rate",
            "depression_dalys_rate": "Depression DALYs rate",
            "addiction_death_rate": "Addiction deaths rate",
            "selfharm_death_rate": "Self-harm deaths rate",
        }
        metric = st.selectbox(
            "Benchmark metric",
            list(feature_labels.keys()),
            format_func=lambda x: feature_labels.get(x, x),
        )
        bench_row = benchmarks[benchmarks["feature"] == metric]
        if not bench_row.empty:
            p10 = float(bench_row["p10"].iloc[0])
            p90 = float(bench_row["p90"].iloc[0])
            table = subset[
                [
                    "location_name",
                    "region_name",
                    metric,
                ]
            ].copy()
            table = table.rename(columns={metric: "value"}).sort_values("value", ascending=False)

            def highlight(value: float) -> str:
                if value >= p90:
                    return "background-color: #fde2e2; color: #9b1b1b;"
                if value <= p10:
                    return "background-color: #e2f0fd; color: #0b4f7c;"
                return ""

            styled = table.head(25).style.applymap(highlight, subset=["value"])
            st.caption(f"Red = above p90 ({p90:.2f}), Blue = below p10 ({p10:.2f}).")
            st.dataframe(styled, use_container_width=True)
            if table.empty:
                bench_summary = "No data for benchmarking."
            else:
                top_row = table.iloc[0]
                bench_summary = (
                    f"p10={p10:.2f}, p90={p90:.2f}; top: {top_row['location_name']} "
                    f"({fmt_value(top_row['value'])})."
                )
            render_chart_guide_from_map(
                "v2_overview_benchmark",
                V2_CHART_GUIDES,
                summary=bench_summary,
                key_prefix="v2_",
            )


def page_v2_clusters() -> None:
    st.markdown('<div class="section-title">v2 Profile Clusters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">KMeans clustering on synthetic 2023 profiles.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_clusters", V2_PAGE_GUIDES)

    data_path = DATA_CLEAN / "v2_clusters.csv"
    if not data_path.exists():
        st.info("Run src/v2_analytics.py to generate cluster outputs.")
        return

    df = load_csv(data_path)
    df = numeric(
        df,
        ["suicide_rate", "depression_dalys_rate", "addiction_death_rate", "selfharm_death_rate"],
    )

    fig = px.choropleth(
        df,
        locations="iso3",
        color="cluster_label",
        hover_name="location_name",
        title="Cluster map (2023, Both)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)
    counts = df["cluster_label"].value_counts().reset_index()
    counts.columns = ["cluster_label", "count"]
    render_chart_guide_from_map(
        "v2_clusters_map",
        V2_CHART_GUIDES,
        summary=summarize_cluster_counts(counts),
        key_prefix="v2_",
    )

    centers = load_report_csv(REPORT_DIR / "v2_cluster_centers.csv")
    if centers is not None and not centers.empty:
        center_cols = [
            "cluster_label",
            "count",
            "suicide_rate",
            "depression_dalys_rate",
            "addiction_death_rate",
            "selfharm_death_rate",
        ]
        st.markdown("### Cluster centers (original units)")
        st.dataframe(centers[[c for c in center_cols if c in centers.columns]], use_container_width=True)
        render_chart_guide_from_map(
            "v2_clusters_centers",
            V2_CHART_GUIDES,
            summary="Centers show average values for each cluster.",
            key_prefix="v2_",
        )

    k_metrics = load_report_csv(REPORT_DIR / "v2_k_selection.csv")
    if k_metrics is not None and not k_metrics.empty:
        st.markdown("### K selection (silhouette)")
        st.dataframe(k_metrics, use_container_width=True)
        if "silhouette" in k_metrics.columns and "k" in k_metrics.columns:
            best = k_metrics.loc[k_metrics["silhouette"].idxmax()]
            k_summary = f"Best silhouette at k={int(best['k'])} ({best['silhouette']:.2f})."
        else:
            k_summary = "Silhouette scores summarize cluster separation."
        render_chart_guide_from_map(
            "v2_clusters_k",
            V2_CHART_GUIDES,
            summary=k_summary,
            key_prefix="v2_",
        )


def page_v2_trajectory_clusters() -> None:
    st.markdown('<div class="section-title">v2 Trajectory Clusters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Clusters based on 2000–2023 suicide-rate trajectories.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_trajectory", V2_PAGE_GUIDES)

    data_path = DATA_CLEAN / "v2_trajectory_clusters.csv"
    if not data_path.exists():
        st.info("Run src/v2_trajectory.py to generate trajectory outputs.")
        return

    df = load_csv(data_path)
    df = numeric(df, ["slope", "volatility", "peak_value", "last5_change", "mean_rate"])

    fig = px.choropleth(
        df,
        locations="iso3",
        color="cluster_label",
        hover_name="location_name",
        title="Trajectory cluster map (Both sexes)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)
    counts = df["cluster_label"].value_counts().reset_index()
    counts.columns = ["cluster_label", "count"]
    render_chart_guide_from_map(
        "v2_traj_map",
        V2_CHART_GUIDES,
        summary=summarize_cluster_counts(counts),
        key_prefix="v2_",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            df,
            x="slope",
            y="volatility",
            color="cluster_label",
            hover_name="location_name",
            title="Slope vs volatility",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_traj_scatter_slope",
            V2_CHART_GUIDES,
            summary=summarize_corr(df, "slope", "volatility"),
            key_prefix="v2_",
        )
    with col2:
        fig = px.scatter(
            df,
            x="mean_rate",
            y="last5_change",
            color="cluster_label",
            hover_name="location_name",
            title="Mean rate vs last-5y change",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_traj_scatter_mean",
            V2_CHART_GUIDES,
            summary=summarize_corr(df, "mean_rate", "last5_change"),
            key_prefix="v2_",
        )

    centers = load_report_csv(REPORT_DIR / "v2_trajectory_cluster_centers.csv")
    if centers is not None and not centers.empty:
        st.markdown("### Cluster centers (trajectory features)")
        st.dataframe(centers, use_container_width=True)
        render_chart_guide_from_map(
            "v2_traj_centers",
            V2_CHART_GUIDES,
            summary="Centers summarize average trajectory features per cluster.",
            key_prefix="v2_",
        )

    k_metrics = load_report_csv(REPORT_DIR / "v2_trajectory_k_selection.csv")
    if k_metrics is not None and not k_metrics.empty:
        st.markdown("### K selection (silhouette)")
        st.dataframe(k_metrics, use_container_width=True)
        if "silhouette" in k_metrics.columns and "k" in k_metrics.columns:
            best = k_metrics.loc[k_metrics["silhouette"].idxmax()]
            k_summary = f"Best silhouette at k={int(best['k'])} ({best['silhouette']:.2f})."
        else:
            k_summary = "Silhouette scores summarize cluster separation."
        render_chart_guide_from_map(
            "v2_traj_k",
            V2_CHART_GUIDES,
            summary=k_summary,
            key_prefix="v2_",
        )


def page_v2_dtw_clusters() -> None:
    st.markdown('<div class="section-title">v2 DTW Clusters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">DTW clustering on 2000–2023 suicide-rate trajectories.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_dtw", V2_PAGE_GUIDES)

    data_path = REPORT_DIR / "v2_dtw_clusters.csv"
    if not data_path.exists():
        st.info("Run src/v2_dtw_clusters.py to generate DTW cluster outputs.")
        return

    df = load_csv(data_path)
    df = numeric(df, ["mean_rate"])

    fig = px.choropleth(
        df,
        locations="iso3",
        color="cluster_label",
        hover_name="location_name",
        title="DTW cluster map (Both sexes)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)
    counts = df["cluster_label"].value_counts().reset_index()
    counts.columns = ["cluster_label", "count"]
    render_chart_guide_from_map(
        "v2_dtw_map",
        V2_CHART_GUIDES,
        summary=summarize_cluster_counts(counts),
        key_prefix="v2_",
    )

    centers = load_report_csv(REPORT_DIR / "v2_dtw_cluster_centers.csv")
    if centers is not None and not centers.empty:
        year_cols = [col for col in centers.columns if col.isdigit()]
        cluster_choice = st.selectbox(
            "Cluster",
            centers["cluster_label"].dropna().unique().tolist(),
        )
        center = centers[centers["cluster_label"] == cluster_choice]
        if not center.empty and year_cols:
            trend = center.melt(
                id_vars=["cluster", "cluster_label"],
                value_vars=year_cols,
                var_name="year",
                value_name="suicide_rate",
            )
            trend["year"] = pd.to_numeric(trend["year"], errors="coerce")
            trend["suicide_rate"] = pd.to_numeric(trend["suicide_rate"], errors="coerce")
            trend = trend.sort_values("year")
            fig = px.line(
                trend,
                x="year",
                y="suicide_rate",
                title=f"{cluster_choice} prototype trend",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
            render_chart_guide_from_map(
                "v2_dtw_prototype",
                V2_CHART_GUIDES,
                summary=summarize_change(trend, "suicide_rate"),
                key_prefix="v2_",
            )

    k_metrics = load_report_csv(REPORT_DIR / "v2_dtw_k_selection.csv")
    if k_metrics is not None and not k_metrics.empty:
        st.markdown("### K selection (silhouette/inertia)")
        st.dataframe(k_metrics, use_container_width=True)
        if "silhouette" in k_metrics.columns and "k" in k_metrics.columns:
            best = k_metrics.loc[k_metrics["silhouette"].idxmax()]
            k_summary = f"Best silhouette at k={int(best['k'])} ({best['silhouette']:.2f})."
        else:
            k_summary = "Silhouette and inertia summarize DTW clustering fit."
        render_chart_guide_from_map(
            "v2_dtw_k",
            V2_CHART_GUIDES,
            summary=k_summary,
            key_prefix="v2_",
        )


def page_v2_country_network() -> None:
    st.markdown('<div class="section-title">v2 Country Network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Similarity graph on 2023 profiles (cosine). Communities via modularity.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_network", V2_PAGE_GUIDES)

    cluster_path = REPORT_DIR / "v2_graph_clusters.csv"
    edges_path = REPORT_DIR / "v2_graph_edges.csv"
    if not cluster_path.exists() or not edges_path.exists():
        st.info("Run src/v2_graph_cluster.py to generate network outputs.")
        return

    clusters = load_csv(cluster_path)
    clusters = numeric(
        clusters,
        [
            "suicide_rate",
            "degree_centrality",
            "betweenness_centrality",
            "x",
            "y",
        ],
    )
    edges = load_csv(edges_path)
    edges = numeric(edges, ["similarity", "source_x", "source_y", "target_x", "target_y"])

    edge_x, edge_y = [], []
    for _, row in edges.iterrows():
        edge_x.extend([row["source_x"], row["target_x"], None])
        edge_y.extend([row["source_y"], row["target_y"], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(31, 111, 139, 0.2)", width=1),
            hoverinfo="none",
            name="Edges",
        )
    )

    clusters = clusters.copy()
    clusters["node_size"] = (clusters["degree_centrality"] * 35 + 6).clip(6, 30)
    palette = px.colors.qualitative.Set2
    labels = sorted(clusters["cluster_label"].dropna().unique().tolist())
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(labels)}

    for label, group in clusters.groupby("cluster_label"):
        fig.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                marker=dict(
                    size=group["node_size"],
                    color=color_map.get(label, palette[0]),
                    line=dict(width=0.5, color="white"),
                ),
                text=group["location_name"],
                customdata=group[["degree_centrality", "betweenness_centrality"]].to_numpy(),
                hovertemplate=(
                    f"<b>%{{text}}</b><br>Cluster={label}<br>Degree=%{{customdata[0]:.2f}}"
                    f"<br>Betweenness=%{{customdata[1]:.2f}}<extra></extra>"
                ),
                name=str(label),
            )
        )
    fig.update_layout(
        title="Country similarity network (2023, Both sexes)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    network_summary = (
        f"{clusters['location_name'].nunique():,} nodes, {len(edges):,} edges, "
        f"{clusters['cluster_label'].nunique():,} communities."
    )
    render_chart_guide_from_map(
        "v2_network_plot",
        V2_CHART_GUIDES,
        summary=network_summary,
        key_prefix="v2_",
    )

    st.markdown("### Community sizes")
    size_df = clusters.groupby("cluster_label", as_index=False).size()
    st.dataframe(size_df.sort_values("size", ascending=False), use_container_width=True)
    size_counts = size_df.rename(columns={"size": "count"})
    render_chart_guide_from_map(
        "v2_network_sizes",
        V2_CHART_GUIDES,
        summary=summarize_cluster_counts(size_counts.sort_values("count", ascending=False)),
        key_prefix="v2_",
    )

    st.markdown("### Top central countries")
    top_central = clusters.sort_values("betweenness_centrality", ascending=False).head(15)
    st.dataframe(
        top_central[
            [
                "location_name",
                "region_name",
                "cluster_label",
                "degree_centrality",
                "betweenness_centrality",
            ]
        ],
        use_container_width=True,
    )
    if top_central.empty:
        central_summary = "No centrality values available."
    else:
        top_row = top_central.iloc[0]
        central_summary = (
            f"Top: {top_row['location_name']} (betweenness {top_row['betweenness_centrality']:.2f})."
        )
    render_chart_guide_from_map(
        "v2_network_central",
        V2_CHART_GUIDES,
        summary=central_summary,
        key_prefix="v2_",
    )

    st.markdown("### Strongest edges")
    st.dataframe(
        edges[["source_name", "target_name", "similarity"]].head(20),
        use_container_width=True,
    )
    if edges.empty:
        edge_summary = "No edges available."
    else:
        top_edge = edges.sort_values("similarity", ascending=False).iloc[0]
        edge_summary = (
            f"Top edge: {top_edge['source_name']} - {top_edge['target_name']} "
            f"(similarity {top_edge['similarity']:.2f})."
        )
    render_chart_guide_from_map(
        "v2_network_edges",
        V2_CHART_GUIDES,
        summary=edge_summary,
        key_prefix="v2_",
    )


def page_v2_linked_views() -> None:
    st.markdown('<div class="section-title">v2 Linked Views</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Brush the scatter to filter map + table. Small multiples for trends.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_linked", V2_PAGE_GUIDES)

    data_path = DATA_CLEAN / "synth_country_year.csv"
    region_path = DATA_CLEAN / "synth_region_year.csv"
    if not data_path.exists() or not region_path.exists():
        st.info("Run src/v2_generate_synth.py to generate v2 datasets.")
        return

    df = load_csv(data_path)
    df = numeric(
        df,
        [
            "year",
            "suicide_rate",
            "depression_dalys_rate",
            "addiction_death_rate",
            "selfharm_death_rate",
        ],
    )
    df = df[df["year"].notna()]

    year = st.selectbox("Year", sorted(df["year"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex_name"].unique()))
    subset = df[(df["year"] == year) & (df["sex_name"] == sex)]

    feature_cols = [
        "suicide_rate",
        "depression_dalys_rate",
        "addiction_death_rate",
        "selfharm_death_rate",
    ]
    x_col = st.selectbox("X axis", feature_cols, index=1)
    y_col = st.selectbox("Y axis", feature_cols, index=0)

    subset = subset.dropna(subset=[x_col, y_col])
    if subset.empty:
        st.info("No data available for this selection. Try another year or sex.")
        return

    scatter = px.scatter(
        subset,
        x=x_col,
        y=y_col,
        color="region_name",
        hover_name="location_name",
        custom_data=["iso3"],
        title="Select countries (box or lasso)",
    )
    scatter.update_layout(dragmode="lasso", height=520)
    scatter.update_traces(marker=dict(size=7, opacity=0.85))

    enable_brush = st.checkbox("Enable brush selection", value=True)
    if enable_brush:
        event = st.plotly_chart(
            scatter,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("box", "lasso"),
            key=f"linked_{year}_{sex}_{x_col}_{y_col}",
        )
        selection = getattr(event, "selection", {}) or {}
        selected = selection.get("points", []) or []
    else:
        st.plotly_chart(scatter, use_container_width=True)
        selected = []

    selected_iso3 = {
        point["customdata"][0]
        for point in selected
        if isinstance(point.get("customdata"), (list, tuple)) and point.get("customdata")
    }
    if selected_iso3:
        filtered = subset[subset["iso3"].isin(selected_iso3)].copy()
        st.caption(f"Selection: {len(filtered)} countries.")
    else:
        filtered = subset.copy()
        st.caption("Tip: brush the scatter to filter the map and table.")
    scatter_summary = f"Selection size: {len(filtered)} countries."
    render_chart_guide_from_map(
        "v2_linked_scatter",
        V2_CHART_GUIDES,
        summary=scatter_summary,
        key_prefix="v2_",
    )

    fig_map = px.choropleth(
        filtered,
        locations="iso3",
        color=y_col,
        hover_name="location_name",
        color_continuous_scale="Reds",
        title=f"{y_col} (filtered)",
    )
    st.plotly_chart(fig_map, use_container_width=True)
    render_chart_guide_from_map(
        "v2_linked_map",
        V2_CHART_GUIDES,
        summary=summarize_range(filtered, y_col),
        key_prefix="v2_",
    )

    cols = [
        "location_name",
        "region_name",
        "income_group",
        x_col,
        y_col,
    ]
    cols = [col for col in cols if col in filtered.columns]
    st.dataframe(
        filtered.sort_values(y_col, ascending=False)[cols].head(25),
        use_container_width=True,
    )
    if filtered.empty:
        table_summary = "No countries in the current selection."
    else:
        top_row = filtered.sort_values(y_col, ascending=False).iloc[0]
        table_summary = f"Top: {top_row['location_name']} ({fmt_value(top_row[y_col])})."
    render_chart_guide_from_map(
        "v2_linked_table",
        V2_CHART_GUIDES,
        summary=table_summary,
        key_prefix="v2_",
    )

    st.markdown("### Small multiples (regional trends)")
    region_df = load_csv(region_path)
    region_df = numeric(region_df, ["year"] + feature_cols)
    region_df = region_df[region_df["year"].notna()]
    region_sex = st.selectbox(
        "Sex (multiples)",
        sorted(region_df["sex_name"].unique()),
        key="linked_views_sex",
    )
    metric = st.selectbox(
        "Metric (multiples)",
        feature_cols,
        index=0,
        key="linked_views_metric",
    )
    region_subset = region_df[region_df["sex_name"] == region_sex]
    fig = px.line(
        region_subset,
        x="year",
        y=metric,
        facet_col="region_name",
        facet_col_wrap=3,
        title=f"{metric} by region ({region_sex})",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    st.plotly_chart(fig, use_container_width=True)
    region_count = region_subset["region_name"].nunique() if not region_subset.empty else 0
    render_chart_guide_from_map(
        "v2_linked_multiples",
        V2_CHART_GUIDES,
        summary=f"Showing {region_count} regions for {region_sex}.",
        key_prefix="v2_",
    )


def page_v2_forecasts() -> None:
    st.markdown('<div class="section-title">v2 Forecasts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Regional suicide-rate forecasts from synthetic data.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_forecasts", V2_PAGE_GUIDES)

    model_choice = st.radio("Forecast model", ["Classical (linear)", "DL (GRU/LSTM)"])
    if model_choice.startswith("DL"):
        data_path = REPORT_DIR / "v2_dl_forecast_region.csv"
        metrics_path = REPORT_DIR / "v2_dl_metrics.csv"
        if not data_path.exists():
            st.info("Run src/v2_dl_forecast.py to generate DL forecasts.")
            return
    else:
        data_path = REPORT_DIR / "v2_forecast_region.csv"
        metrics_path = None
        if not data_path.exists():
            st.info("Run src/v2_analytics.py to generate forecast outputs.")
            return

    df = load_csv(data_path)
    df = numeric(df, ["suicide_rate", "year"])
    region = st.selectbox("Region", sorted(df["region_name"].unique()))
    subset = df[df["region_name"] == region]
    fig = px.line(
        subset,
        x="year",
        y="suicide_rate",
        color="type",
        markers=True,
        title=f"{region}: actual vs forecast",
    )
    st.plotly_chart(fig, use_container_width=True)
    if "type" in subset.columns:
        actual = subset[subset["type"] == "actual"]
        forecast_summary = summarize_change(actual, "suicide_rate") if not actual.empty else None
    else:
        forecast_summary = summarize_change(subset, "suicide_rate")
    render_chart_guide_from_map(
        "v2_forecast_line",
        V2_CHART_GUIDES,
        summary=forecast_summary,
        key_prefix="v2_",
    )

    if metrics_path and metrics_path.exists():
        metrics = load_csv(metrics_path)
        st.markdown("### DL forecast metrics")
        st.dataframe(metrics, use_container_width=True)
        render_chart_guide_from_map(
            "v2_forecast_metrics",
            V2_CHART_GUIDES,
            summary="Metrics summarize DL forecast error.",
            key_prefix="v2_",
        )


def page_v2_backtest() -> None:
    st.markdown('<div class="section-title">v2 Backtest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Rolling-origin backtest using lag features.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_backtest", V2_PAGE_GUIDES)

    pred_path = REPORT_DIR / "v2_backtest_predictions.csv"
    metrics_path = REPORT_DIR / "v2_backtest_metrics.csv"
    if not pred_path.exists() or not metrics_path.exists():
        st.info("Run src/v2_backtest.py to generate backtest outputs.")
        return

    preds = load_csv(pred_path)
    if preds.empty or not {"region_name", "year", "actual", "predicted"}.issubset(preds.columns):
        st.info("Backtest predictions are empty. Re-run src/v2_backtest.py to regenerate.")
        return
    preds = numeric(preds, ["actual", "predicted", "year"])
    region = st.selectbox("Region", sorted(preds["region_name"].unique()))
    subset = preds[preds["region_name"] == region]
    fig = px.line(
        subset,
        x="year",
        y=["actual", "predicted"],
        markers=True,
        title=f"{region}: actual vs predicted",
    )
    st.plotly_chart(fig, use_container_width=True)
    if subset.empty:
        backtest_summary = "No data for selected region."
    else:
        last_row = subset.sort_values("year").iloc[-1]
        delta = last_row["predicted"] - last_row["actual"]
        backtest_summary = f"Latest gap: {delta:+.2f} (pred - actual)."
    render_chart_guide_from_map(
        "v2_backtest_line",
        V2_CHART_GUIDES,
        summary=backtest_summary,
        key_prefix="v2_",
    )

    metrics = load_csv(metrics_path)
    st.markdown("### Backtest metrics")
    st.dataframe(metrics, use_container_width=True)
    if "mae" in metrics.columns:
        best = metrics.sort_values("mae").iloc[0]
        metrics_summary = f"Best MAE: {best['region_name']} ({best['mae']:.2f})."
    else:
        metrics_summary = "Backtest metrics summary."
    render_chart_guide_from_map(
        "v2_backtest_metrics",
        V2_CHART_GUIDES,
        summary=metrics_summary,
        key_prefix="v2_",
    )


def page_v2_scenario() -> None:
    st.markdown('<div class="section-title">v2 Scenario Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">What-if simulator using synthetic regression model.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_scenario", V2_PAGE_GUIDES)

    coeff_path = REPORT_DIR / "v2_model_coeffs.csv"
    data_path = DATA_CLEAN / "synth_country_year.csv"
    if not coeff_path.exists() or not data_path.exists():
        st.info("Run src/v2_analytics.py to generate model coefficients.")
        return

    coeffs = load_csv(coeff_path)
    coeffs["coef"] = pd.to_numeric(coeffs["coef"], errors="coerce")
    intercept = float(coeffs[coeffs["feature"] == "_intercept"]["coef"].iloc[0])
    coeffs = coeffs[coeffs["feature"] != "_intercept"].copy()
    coeffs["mean"] = pd.to_numeric(coeffs["mean"], errors="coerce")
    coeffs["scale"] = pd.to_numeric(coeffs["scale"], errors="coerce")

    df = load_csv(data_path)
    df = numeric(
        df,
        ["suicide_rate", "depression_dalys_rate", "addiction_death_rate", "selfharm_death_rate"],
    )
    baseline = df[(df["year"] == df["year"].max()) & (df["sex_name"] == "Both")]
    if baseline.empty:
        st.info("No baseline data available for Scenario Lab.")
        return

    feature_cols = ["depression_dalys_rate", "addiction_death_rate", "selfharm_death_rate"]
    coeffs = coeffs[coeffs["feature"].isin(feature_cols)].copy()
    quantiles = baseline[feature_cols].quantile([0.1, 0.5, 0.9])

    inputs = {}
    for feature in coeffs["feature"].tolist():
        q10 = float(quantiles.loc[0.1, feature])
        q50 = float(quantiles.loc[0.5, feature])
        q90 = float(quantiles.loc[0.9, feature])
        min_val = max(0.0, q10 * 0.8)
        max_val = max(min_val + 1.0, q90 * 1.2)
        default_val = q50
        inputs[feature] = st.slider(
            feature,
            min_val,
            max_val,
            default_val,
        )

    pred = intercept
    for _, row in coeffs.iterrows():
        feature = row["feature"]
        mean = float(row["mean"])
        scale = float(row["scale"]) if float(row["scale"]) != 0 else 1.0
        scaled = (inputs[feature] - mean) / scale
        pred += float(row["coef"]) * scaled

    pred = max(0.0, pred)
    st.markdown(f"### Predicted suicide rate: **{pred:.2f}**")
    st.caption("Inputs are bounded to the 10th–90th percentile of recent synthetic data.")
    render_chart_guide_from_map(
        "v2_scenario_pred",
        V2_CHART_GUIDES,
        summary=f"Predicted rate: {pred:.2f} per 100k.",
        key_prefix="v2_",
    )

    metrics_path = REPORT_DIR / "v2_model_metrics.csv"
    if metrics_path.exists():
        st.markdown("### Model metrics")
        st.dataframe(load_csv(metrics_path), use_container_width=True)

    sens_path = REPORT_DIR / "v2_sensitivity.csv"
    if sens_path.exists():
        st.markdown("### Sensitivity (10% increase per feature)")
        sens = load_csv(sens_path)
        sens = numeric(sens, ["pct_change_prediction"])
        fig = px.bar(
            sens,
            x="pct_change_prediction",
            y="feature",
            orientation="h",
            title="Elasticity on predicted suicide rate",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_sensitivity",
            V2_CHART_GUIDES,
            summary="Bars show percent change in prediction for a 10% input change.",
            key_prefix="v2_",
        )

    quant_path = REPORT_DIR / "v2_quantile_predictions.csv"
    if quant_path.exists():
        st.markdown("### Prediction intervals (quantile regression)")
        st.caption("Bands show q10–q90 intervals; q50 is the median prediction.")
        quant = load_csv(quant_path)
        quant = numeric(quant, ["year", "suicide_rate", "q10", "q50", "q90"])
        region_opt = sorted(quant["region_name"].dropna().unique())
        sex_opt = sorted(quant["sex_name"].dropna().unique())
        region = st.selectbox("Region (intervals)", region_opt, key="quant_region")
        sex = st.selectbox("Sex (intervals)", sex_opt, key="quant_sex")
        subset = quant[(quant["region_name"] == region) & (quant["sex_name"] == sex)]
        if subset.empty:
            st.info("No quantile predictions for this selection.")
            return
        grouped = (
            subset.groupby("year", as_index=False)[["suicide_rate", "q10", "q50", "q90"]].mean()
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grouped["year"],
                y=grouped["q90"],
                line=dict(color="rgba(31, 111, 139, 0.2)"),
                name="q90",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["year"],
                y=grouped["q10"],
                fill="tonexty",
                fillcolor="rgba(31, 111, 139, 0.2)",
                line=dict(color="rgba(31, 111, 139, 0.2)"),
                name="q10-q90 band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["year"],
                y=grouped["q50"],
                line=dict(color="#1f6f8b"),
                name="q50",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["year"],
                y=grouped["suicide_rate"],
                line=dict(color="#f2b950"),
                name="actual",
            )
        )
        fig.update_layout(
            title=f"{region} quantile predictions (avg across countries)",
            xaxis_title="Year",
            yaxis_title="Suicide rate",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_quantile",
            V2_CHART_GUIDES,
            summary=summarize_change(grouped, "suicide_rate"),
            key_prefix="v2_",
        )

    metrics_path = REPORT_DIR / "v2_quantile_metrics.csv"
    if metrics_path.exists():
        st.markdown("### Quantile model metrics")
        st.dataframe(load_csv(metrics_path), use_container_width=True)
        render_chart_guide_from_map(
            "v2_quantile_metrics",
            V2_CHART_GUIDES,
            summary="Quantile metrics summarize interval performance.",
            key_prefix="v2_",
        )

    perm_path = REPORT_DIR / "v2_perm_importance.csv"
    pdp_path = REPORT_DIR / "v2_partial_dependence.csv"
    if perm_path.exists() and pdp_path.exists():
        st.markdown("### Explainability")
        perm = load_csv(perm_path)
        perm = numeric(perm, ["importance_mean", "importance_std"])
        fig = px.bar(
            perm.sort_values("importance_mean", ascending=True),
            x="importance_mean",
            y="feature",
            orientation="h",
            title="Permutation importance (R2 drop)",
        )
        st.plotly_chart(fig, use_container_width=True)
        if perm.empty:
            perm_summary = "No permutation results available."
        else:
            top_feat = perm.sort_values("importance_mean", ascending=False).iloc[0]
            perm_summary = f"Top feature: {top_feat['feature']} ({top_feat['importance_mean']:.3f})."
        render_chart_guide_from_map(
            "v2_explain_perm",
            V2_CHART_GUIDES,
            summary=perm_summary,
            key_prefix="v2_",
        )

        pdp = load_csv(pdp_path)
        pdp = numeric(pdp, ["feature_value", "pdp"])
        fig = px.line(
            pdp,
            x="feature_value",
            y="pdp",
            color="feature",
            title="Partial dependence (top 3 features)",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_chart_guide_from_map(
            "v2_explain_pdp",
            V2_CHART_GUIDES,
            summary="Lines show marginal effects by feature value.",
            key_prefix="v2_",
        )


def page_v2_outliers() -> None:
    st.markdown('<div class="section-title">v2 Outliers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">IsolationForest anomalies (2023 profiles).</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_outliers", V2_PAGE_GUIDES)

    data_path = REPORT_DIR / "v2_outliers.csv"
    if not data_path.exists():
        st.info("Run src/v2_analytics.py to generate outlier outputs.")
        return

    df = load_csv(data_path)
    df = numeric(
        df,
        [
            "suicide_rate",
            "depression_dalys_rate",
            "addiction_death_rate",
            "selfharm_death_rate",
            "outlier_score",
        ],
    )
    df["outlier_score_size"] = df["outlier_score"].clip(lower=0)

    fig = px.scatter(
        df,
        x="depression_dalys_rate",
        y="suicide_rate",
        color="is_outlier",
        size="outlier_score_size",
        hover_name="location_name",
        title="Outliers in synthetic feature space",
    )
    st.plotly_chart(fig, use_container_width=True)
    outlier_count = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0
    render_chart_guide_from_map(
        "v2_outliers_scatter",
        V2_CHART_GUIDES,
        summary=summarize_outlier_counts(len(df), outlier_count),
        key_prefix="v2_",
    )

    top = df.sort_values("outlier_score", ascending=False).head(20)
    st.markdown("### Top anomalies")
    st.dataframe(
        top[
            [
                "location_name",
                "region_name",
                "outlier_score",
                "outlier_reason",
                "suicide_rate",
                "depression_dalys_rate",
                "addiction_death_rate",
                "selfharm_death_rate",
            ]
        ],
        use_container_width=True,
    )
    if top.empty:
        top_summary = "No outliers available."
    else:
        top_row = top.iloc[0]
        top_summary = f"Top: {top_row['location_name']} (score {top_row['outlier_score']:.2f})."
    render_chart_guide_from_map(
        "v2_outliers_table",
        V2_CHART_GUIDES,
        summary=top_summary,
        key_prefix="v2_",
    )


def page_v2_patterns() -> None:
    st.markdown('<div class="section-title">v2 Patterns</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Association rules on binned 2023 profiles.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide_from_map("v2_patterns", V2_PAGE_GUIDES)

    rules_path = REPORT_DIR / "v2_assoc_rules.csv"
    if not rules_path.exists():
        st.info("Run src/v2_assoc_rules.py to generate association rules.")
        return

    rules = load_csv(rules_path)
    rules = numeric(rules, ["support", "confidence", "lift"])
    st.markdown("### Top rules (by lift)")
    st.dataframe(rules.head(20), use_container_width=True)
    if rules.empty:
        rules_summary = "No association rules available."
    else:
        row = rules.iloc[0]
        rules_summary = f"Top lift: {row['antecedents']} -> {row['consequents']} (lift {row['lift']:.2f})."
    render_chart_guide_from_map(
        "v2_patterns_table",
        V2_CHART_GUIDES,
        summary=rules_summary,
        key_prefix="v2_",
    )

    st.markdown("### Rule interpretation")
    if not rules.empty:
        row = rules.iloc[0]
        st.info(
            f"When {row['antecedents']} then {row['consequents']} "
            f"(lift {row['lift']:.2f}, confidence {row['confidence']:.2f})."
        )
        interp_summary = (
            f"Lift {row['lift']:.2f}, confidence {row['confidence']:.2f} for top rule."
        )
        render_chart_guide_from_map(
            "v2_patterns_interp",
            V2_CHART_GUIDES,
            summary=interp_summary,
            key_prefix="v2_",
        )


def page_v3_risk_estimator() -> None:
    st.markdown('<div class="section-title">v3 Risk Estimator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Interactive probability of high-risk suicide category.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This is a demo model: probabilities reflect patterns in the selected dataset, not clinical risk."
    )
    with st.expander("How to read this page", expanded=False):
        st.markdown(
            """
- **High-risk cutoff** sets the percentile threshold for labeling high-risk.
- **Predicted probability** is the model's confidence for high-risk.
- **Calibration plot** checks if predicted probabilities match observed rates.
- **Counterfactual hints** show sensitivity to a 10% reduction per feature.
- **Local contributions** show which inputs push the prediction up or down.
""".strip()
        )

    if ColumnTransformer is None or LogisticRegression is None:
        st.info("Install scikit-learn to enable the v3 estimator.")
        return

    source = st.selectbox("Training data source", ["v1", "v2"], index=0)
    df = load_v3_features(source)
    if df is None or df.empty:
        st.info("Run python scripts/run_v3_pipeline.py to generate v3 feature tables.")
        return

    sex_options = sorted(df["sex_name"].dropna().unique()) if "sex_name" in df.columns else ["Both"]
    sex = st.selectbox("Sex", sex_options)
    df_sex = df[df["sex_name"] == sex] if "sex_name" in df.columns else df
    if df_sex.empty:
        df_sex = df

    country = st.selectbox("Country", sorted(df_sex["location_name"].unique()))
    country_row = df_sex[df_sex["location_name"] == country].iloc[0]

    use_defaults = st.toggle("Use country defaults", value=True)
    st.caption(
        f"Region: {country_row['region_name']} | Income group: {country_row['income_group']}"
    )

    def slider_params(series: pd.Series) -> tuple[float, float, float]:
        min_v = float(series.min())
        max_v = float(series.max())
        if min_v == max_v:
            max_v = min_v + 1.0
        span = max_v - min_v
        step = min(max(span / 100, 0.1), 5.0)
        return min_v, max_v, round(step, 2)

    def default_value(col: str) -> float:
        value = float(country_row[col]) if use_defaults and pd.notna(country_row[col]) else float(df[col].median())
        return max(float(df[col].min()), min(value, float(df[col].max())))

    col1, col2 = st.columns(2)
    with col1:
        min_v, max_v, step = slider_params(df["depression_dalys_rate"])
        depression = st.slider(
            "Depression DALYs rate",
            min_value=min_v,
            max_value=max_v,
            value=default_value("depression_dalys_rate"),
            step=step,
        )
        min_v, max_v, step = slider_params(df["addiction_death_rate"])
        addiction = st.slider(
            "Addiction death rate",
            min_value=min_v,
            max_value=max_v,
            value=default_value("addiction_death_rate"),
            step=step,
        )
    with col2:
        min_v, max_v, step = slider_params(df["selfharm_death_rate"])
        selfharm = st.slider(
            "Self-harm death rate",
            min_value=min_v,
            max_value=max_v,
            value=default_value("selfharm_death_rate"),
            step=step,
        )
        cutoff_pct = st.slider(
            "High-risk cutoff percentile",
            min_value=60,
            max_value=90,
            value=80,
            step=5,
        )

    cutoff = cutoff_pct / 100
    base_model, calibrated_model, metrics, cat_cols, num_cols = train_v3_model(df, cutoff)
    predictor = calibrated_model if calibrated_model is not None else base_model

    input_row = {
        "iso3": country_row.get("iso3", ""),
        "location_name": country_row.get("location_name", ""),
        "region_name": country_row.get("region_name", ""),
        "income_group": country_row.get("income_group", ""),
        "sex_name": sex if "sex_name" in df.columns else "Both",
        "depression_dalys_rate": float(depression),
        "addiction_death_rate": float(addiction),
        "selfharm_death_rate": float(selfharm),
    }
    input_df = pd.DataFrame([input_row])[cat_cols + num_cols]
    proba = float(predictor.predict_proba(input_df)[0][1])

    rate_threshold = metrics["threshold_rate"]
    st.caption(
        f"High-risk means suicide_rate ≥ {rate_threshold:.2f} per 100k "
        f"(p{cutoff_pct} of the selected dataset)."
    )
    if calibrated_model is not None:
        st.caption("Calibration: isotonic (cv=3).")
    else:
        st.caption("Calibration: off (insufficient data or sklearn unavailable).")

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.metric("Predicted high-risk probability", f"{proba:.1%}")
    with kpi_cols[1]:
        label = "High-risk" if proba >= 0.5 else "Not high-risk"
        st.metric("Predicted label", label)
    with kpi_cols[2]:
        st.metric("Baseline high-risk rate", f"{metrics['positive_rate']:.1%}")

    if "train_accuracy" in metrics or "train_auc" in metrics:
        st.markdown("### Model diagnostics (training set)")
        st.caption("Training-only metrics; use as a quick sanity check.")
        diag_items = []
        if "train_accuracy" in metrics:
            diag_items.append(("Accuracy", f"{metrics['train_accuracy']:.2f}"))
        if "train_auc" in metrics:
            diag_items.append(("ROC AUC", f"{metrics['train_auc']:.2f}"))
        if "train_brier" in metrics:
            diag_items.append(("Brier (lower=better)", f"{metrics['train_brier']:.3f}"))
        render_kpis(diag_items)

    st.markdown("### Calibration reliability")
    st.caption("Closer to the diagonal means probabilities match observed rates.")
    proba_all = predictor.predict_proba(df[cat_cols + num_cols])[:, 1]
    y_all = (df["suicide_rate"] >= rate_threshold).astype(int)
    rel = v3_reliability_curve(proba_all, y_all, bins=8)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rel["mean_pred"],
            y=rel["observed_rate"],
            mode="markers+lines",
            name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal",
            line=dict(dash="dash", color="#6b6460"),
        )
    )
    fig.update_layout(
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed rate",
        title="Reliability plot (binned)",
    )
    st.plotly_chart(fig, use_container_width=True)
    if "train_brier" in metrics:
        st.caption(f"Brier score (train): {metrics['train_brier']:.3f}")

    st.markdown("### Counterfactual hints (10% reduction)")
    st.caption("Not causal; shows local sensitivity if one feature is reduced by 10%.")
    cf_rows = []
    for feature in V3_NUMERIC_COLS:
        new_row = dict(input_row)
        new_val = max(0.0, float(new_row[feature]) * 0.9)
        new_row[feature] = new_val
        new_df = pd.DataFrame([new_row])[cat_cols + num_cols]
        new_proba = float(predictor.predict_proba(new_df)[0][1])
        cf_rows.append(
            {
                "feature": feature.replace("_", " "),
                "new_value": new_val,
                "delta_probability": new_proba - proba,
            }
        )
    cf_df = pd.DataFrame(cf_rows)
    cf_df["delta_probability"] = cf_df["delta_probability"].map(lambda x: round(x, 4))
    st.dataframe(cf_df, use_container_width=True)

    st.markdown("### Local feature contributions")
    st.caption("Top drivers for this prediction in log-odds space (positive increases risk).")
    contrib = v3_feature_contributions(base_model, input_df)
    top = contrib.head(10)
    fig = px.bar(
        top.sort_values("contribution"),
        x="contribution",
        y="feature",
        orientation="h",
        title="Top contributions (log-odds)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Similar countries (nearest neighbors)")
    st.caption("Closest countries in feature space (standardized numeric inputs).")
    neighbors = v3_neighbors(df, input_row, rate_threshold, k=5)
    cols = [
        "location_name",
        "region_name",
        "income_group",
        "suicide_rate",
        "high_risk",
        "distance",
    ]
    cols = [col for col in cols if col in neighbors.columns]
    st.dataframe(neighbors[cols], use_container_width=True)


def page_v3_methods() -> None:
    st.markdown('<div class="section-title">v3 Methods</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Model scope, inputs, and dataset notes.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "The v3 estimator predicts high-risk category membership, not individual outcomes."
    )

    summary_path = REPORT_DIR / "v3_feature_summary.md"
    render_markdown_file(
        summary_path,
        f"Missing {summary_path}. Run src/v3_prepare_features.py.",
        strip_title=True,
    )
def page_relationships() -> None:
    st.markdown('<div class="section-title">Relationships</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Correlation views from the merged ML dataset.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("relationships")
    df = ensure_exists(DATA_CLEAN / "merged_ml_country.csv", "Merged ML dataset")
    df = numeric(df, [col for col in df.columns if col.startswith("gbd_")])
    df = numeric(df, ["age_standardized_suicide_rate_2021"])

    age = st.selectbox("Age group", sorted(df["age_name"].unique()))
    df = df[df["age_name"] == age]

    gbd_cols = [col for col in df.columns if col.startswith("gbd_") and col != "gbd_year"]
    x_col = st.selectbox("X variable", gbd_cols, index=0)
    y_col = st.selectbox("Y variable", ["age_standardized_suicide_rate_2021"] + gbd_cols, index=0)

    scatter = df.dropna(subset=[x_col, y_col])
    fig = px.scatter(
        scatter,
        x=x_col,
        y=y_col,
        color="region_name",
        hover_name="location_name",
        title=f"{y_col} vs {x_col}",
    )
    st.plotly_chart(fig, use_container_width=True)
    render_chart_guide("relationships_scatter", summary=summarize_corr(scatter, x_col, y_col))

    st.markdown("### Correlation heatmap")
    corr_cols = ["age_standardized_suicide_rate_2021"] + gbd_cols
    corr = df[corr_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        title="Correlation matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
    corr_abs = corr.abs().copy()
    np.fill_diagonal(corr_abs.values, 0)
    if corr_abs.values.max() == 0:
        heat_summary = "No non-diagonal correlations available."
    else:
        max_pair = corr_abs.stack().idxmax()
        heat_summary = (
            f"Strongest absolute correlation: {max_pair[0]} vs {max_pair[1]} "
            f"(r={corr.loc[max_pair]:.2f})."
        )
    render_chart_guide("relationships_heatmap", summary=heat_summary)


def page_ml_demo() -> None:
    st.markdown('<div class="section-title">ML Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Ridge and RandomForest quick model comparison.</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("ml_demo")
    if ColumnTransformer is None:
        st.error("scikit-learn is required for the ML demo.")
        return

    df = load_ml_baseline_features()
    required_cols = [
        "age_standardized_suicide_rate_2021",
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_selfharm_death_rate_both",
        "region_name",
        "income_group",
        "data_quality",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            "Missing columns for ML demo: "
            + ", ".join(missing)
            + ". Re-run src/06_ml_baseline.py."
        )
        return

    target_options = ["age_standardized_suicide_rate_2021"]
    if "crude_suicide_rate_2021" in df.columns:
        target_options.append("crude_suicide_rate_2021")

    st.caption("Uses one row per country to avoid leakage across age groups.")
    target = st.selectbox("Target", target_options)
    test_size = st.slider("Test size", 0.2, 0.4, 0.25, step=0.05)

    feature_cols = [
        "gbd_depression_dalys_rate_both",
        "gbd_addiction_death_rate_both",
        "gbd_selfharm_death_rate_both",
    ]
    categorical_features = ["region_name", "income_group", "data_quality"]

    model_df = df.dropna(subset=[target] + feature_cols)
    if model_df.empty:
        st.warning("Not enough data for ML demo.")
        return

    X = model_df[feature_cols + categorical_features]
    y = model_df[target]

    numeric_features = feature_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    ridge = Pipeline(
        steps=[("prep", preprocessor), ("model", Ridge(alpha=1.0))]
    )
    rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    models = {"Ridge": ridge, "RandomForest": rf}
    results = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
        )
        preds[name] = y_pred

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    if results_df.empty:
        results_summary = "No model results available."
    else:
        best = results_df.sort_values("MAE").iloc[0]
        results_summary = (
            f"Best MAE: {best['Model']} "
            f"(MAE {best['MAE']:.2f}, R2 {best['R2']:.2f})."
        )
    render_chart_guide("ml_results", summary=results_summary)

    cv_path = REPORT_DIR / "ml_baseline_cv.csv"
    if cv_path.exists():
        st.markdown("### Cross-validation (5-fold)")
        cv_df = load_csv(cv_path)
        st.dataframe(cv_df, use_container_width=True)
        if cv_df.empty or "mae_mean" not in cv_df.columns:
            cv_summary = "No cross-validation summary available."
        else:
            best_cv = cv_df.sort_values("mae_mean").iloc[0]
            cv_summary = (
                f"Best CV MAE: {best_cv['model']} "
                f"(MAE {best_cv['mae_mean']:.2f} +/- {best_cv['mae_std']:.2f})."
            )
        render_chart_guide("ml_cv", summary=cv_summary)
    else:
        st.info("Run src/06_ml_baseline.py to generate cross-validation metrics.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=preds["Ridge"],
            mode="markers",
            name="Ridge",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=preds["RandomForest"],
            mode="markers",
            name="RandomForest",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_test,
            mode="lines",
            name="Ideal",
        )
    )
    fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig, use_container_width=True)
    if results_df.empty:
        pred_summary = "No model results available."
    else:
        best_r2 = results_df.sort_values("R2", ascending=False).iloc[0]
        pred_summary = f"Top R2: {best_r2['Model']} ({best_r2['R2']:.2f})."
    render_chart_guide("ml_pred_actual", summary=pred_summary)

    st.markdown("### Feature importance (RandomForest)")
    rf_model = rf.named_steps["model"]
    feature_names = (
        rf.named_steps["prep"]
        .get_feature_names_out()
        .tolist()
    )
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    fig = px.bar(
        imp_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 features",
    )
    st.plotly_chart(fig, use_container_width=True)
    if imp_df.empty:
        imp_summary = "No importance values available."
    else:
        imp_summary = f"Top feature: {imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['importance']:.3f})."
    render_chart_guide("ml_feature_importance", summary=imp_summary)


def page_methods() -> None:
    if VERSION == "v2":
        st.markdown(
            '<div class="section-title">Methods & Synthetic Data</div>',
            unsafe_allow_html=True,
        )
        st.info("Synthetic data for demonstration only.")
        render_page_guide_from_map("v2_methods", V2_PAGE_GUIDES)

        st.markdown("### Synthetic generation notes")
        render_markdown_file(
            REPORT_DIR / "synth_generation_notes.md",
            f"Missing {REPORT_DIR / 'synth_generation_notes.md'}. Run src/v2_generate_synth.py.",
            strip_title=True,
        )

        with st.expander("Synthetic data dictionary", expanded=False):
            render_markdown_file(
                REPORT_DIR / "synth_data_dictionary.md",
                f"Missing {REPORT_DIR / 'synth_data_dictionary.md'}. Run src/v2_generate_synth.py.",
                strip_title=True,
            )

        st.markdown("### Synthetic validity report")
        render_markdown_file(
            REPORT_DIR / "v2_validity_report.md",
            f"Missing {REPORT_DIR / 'v2_validity_report.md'}. Run src/v2_validity_report.py.",
            strip_title=True,
        )

        st.markdown("### Analytics notes")
        render_markdown_file(
            REPORT_DIR / "v2_analytics_notes.md",
            f"Missing {REPORT_DIR / 'v2_analytics_notes.md'}. Run src/v2_analytics.py.",
            strip_title=True,
        )

        with st.expander("Advanced methods notes", expanded=False):
            st.markdown("#### Trajectory clustering")
            render_markdown_file(
                REPORT_DIR / "v2_trajectory_notes.md",
                f"Missing {REPORT_DIR / 'v2_trajectory_notes.md'}. Run src/v2_trajectory.py.",
                strip_title=True,
            )
            st.markdown("#### DTW clustering")
            render_markdown_file(
                REPORT_DIR / "v2_dtw_notes.md",
                f"Missing {REPORT_DIR / 'v2_dtw_notes.md'}. Run src/v2_dtw_clusters.py.",
                strip_title=True,
            )
            st.markdown("#### Graph clustering")
            render_markdown_file(
                REPORT_DIR / "v2_graph_notes.md",
                f"Missing {REPORT_DIR / 'v2_graph_notes.md'}. Run src/v2_graph_cluster.py.",
                strip_title=True,
            )
            st.markdown("#### DL forecast")
            render_markdown_file(
                REPORT_DIR / "v2_dl_notes.md",
                f"Missing {REPORT_DIR / 'v2_dl_notes.md'}. Run src/v2_dl_forecast.py.",
                strip_title=True,
            )

        metrics_path = REPORT_DIR / "v2_model_metrics.csv"
        if metrics_path.exists():
            st.markdown("### Model metrics")
            st.dataframe(load_csv(metrics_path), use_container_width=True)

        st.markdown("### Data Quality (Great Expectations)")
        ge_summary = REPORT_DIR / "v2_quality_summary.md"
        render_markdown_file(
            ge_summary,
            f"Missing {ge_summary}. Run src/v2_ge_validate.py.",
            strip_title=True,
        )

        ge_report = REPORT_DIR / "ge_report.html"
        if ge_report.exists():
            with st.expander("Open Great Expectations HTML report", expanded=False):
                st.components.v1.html(ge_report.read_text(encoding="utf-8"), height=520, scrolling=True)
        return

    st.markdown(
        '<div class="section-title">Methods, Data Model & Quality</div>',
        unsafe_allow_html=True,
    )
    render_page_guide("methods")
    st.markdown(
        """
**Definitions and units**
- WHO suicide metrics are 2021 counts and rates per 100,000.
- GBD DALYs/deaths pages use Number / Percent / Rate, while probability-of-death is a 0-1 probability (not per 100k).

**Cross-year merge warning**
- The ML table pairs WHO 2021 outcomes with GBD 2023 features. Interpret as correlational, not causal.

**Ecological fallacy**
- Country-level associations do not imply individual-level relationships.

**Missingness**
- GBD features were filtered to core causes; countries missing ISO3 mappings are excluded.

**Population-weighted aggregates**
- Regional/global context rows use population weights derived from Number/Rate relationships.
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Data Model (Star Schema)")
    render_markdown_file(
        REPORT_DIR / "data_model.md",
        f"Data model not found. Run the BI pack to generate {REPORT_DIR / 'data_model.md'}.",
        strip_title=True,
    )

    with st.expander("Data Dictionary", expanded=False):
        render_markdown_file(
            REPORT_DIR / "data_dictionary.md",
            f"Data dictionary not found. Run the BI pack to generate {REPORT_DIR / 'data_dictionary.md'}.",
            strip_title=True,
        )

    st.markdown("### Data Quality Scorecard")
    scorecard_df = load_report_csv(REPORT_DIR / "data_quality_scorecard.csv")
    if scorecard_df is None or scorecard_df.empty:
        st.info("Data quality scorecard not found. Run src/07_data_quality_scorecard.py.")
    else:
        st.dataframe(scorecard_df, use_container_width=True)

    missingness_df = load_report_csv(REPORT_DIR / "data_quality_missingness.csv")
    if missingness_df is not None and not missingness_df.empty:
        top_missing = missingness_df.sort_values(
            "missing_pct", ascending=False
        ).head(12)
        fig = px.bar(
            top_missing,
            x="missing_pct",
            y="column",
            color="dataset",
            orientation="h",
            title="Top missingness (percentage)",
        )
        st.plotly_chart(fig, use_container_width=True)

    dq_df = load_report_csv(REPORT_DIR / "data_quality_who_data_quality.csv")
    if dq_df is not None and not dq_df.empty:
        fig = px.pie(
            dq_df,
            names="data_quality",
            values="count",
            title="WHO data_quality distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    iso_df = load_report_csv(REPORT_DIR / "data_quality_iso3_unmatched.csv")
    if iso_df is not None and not iso_df.empty:
        st.markdown("#### ISO3 unmatched (by source_type)")
        st.dataframe(iso_df, use_container_width=True)


PAGES_V1 = {
    "Overview / Story": page_overview,
    "WHO Suicide Explorer": page_who_explorer,
    "Depression Burden (GBD)": page_depression,
    "Addiction (GBD)": page_addiction,
    "Self-harm (GBD)": page_selfharm,
    "Probability of Death (GBD)": page_prob_death,
    "All-cause Trends": page_allcause_trends,
    "Big Categories": page_big_categories,
    "Relationships": page_relationships,
    "Country Segmentation": page_segmentation,
    "Outliers & Alerts": page_outliers,
    "ML Demo": page_ml_demo,
    "Methods + Data Model + Quality": page_methods,
}

PAGES_V2 = {
    "v2 Overview": page_v2_overview,
    "v2 Clusters": page_v2_clusters,
    "v2 Trajectory Clusters": page_v2_trajectory_clusters,
    "v2 DTW Clusters": page_v2_dtw_clusters,
    "v2 Country Network": page_v2_country_network,
    "v2 Linked Views": page_v2_linked_views,
    "v2 Forecasts": page_v2_forecasts,
    "v2 Backtest": page_v2_backtest,
    "v2 Scenario Lab": page_v2_scenario,
    "v2 Outliers": page_v2_outliers,
    "v2 Patterns": page_v2_patterns,
    "Methods + Data Model + Quality": page_methods,
}

PAGES_V3 = {
    "v3 Risk Estimator": page_v3_risk_estimator,
    "v3 Methods": page_v3_methods,
}

PAGES_V0 = {
    "v0 Static Gallery": page_v0_gallery,
}

if VERSION == "v2":
    PAGES = PAGES_V2
elif VERSION == "v3":
    PAGES = PAGES_V3
elif VERSION == "v0":
    PAGES = PAGES_V0
else:
    PAGES = PAGES_V1


def main() -> None:
    st.sidebar.markdown("## Navigation")
    st.sidebar.markdown(f"**Version**: {VERSION}")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.markdown('<span class="info-chip">Layer A + Layer B integrated</span>', unsafe_allow_html=True)
    PAGES[page]()


if __name__ == "__main__":
    main()
