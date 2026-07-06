// Curated knowledge base for the guide chatbot.
// Answers are derived from the project's RESULTS only (ML / DL / DM outputs and
// model predictions). No raw data is exposed. Fully client-side, no LLM.

export const INTENTS = [
  {
    id: "drivers",
    keywords: ["driver", "drive", "cause", "factor", "risk", "shap", "important", "influence", "correlate", "why", "raise", "increase"],
    answer:
      "The strongest independent drivers of the national suicide rate (from SHAP on the leakage-free model) are life expectancy and alcohol consumption per capita, followed by addiction burden, GDP per capita, and urbanization. Lower life expectancy and higher alcohol use push predicted risk up — both are well-established epidemiological signals.",
    followups: ["How accurate is the model?", "Show high-risk country groups"],
  },
  {
    id: "accuracy",
    keywords: ["accurate", "accuracy", "r2", "r²", "mae", "error", "performance", "good", "reliable", "score", "metric", "how well"],
    answer:
      "Honest, cross-validated numbers: on independent predictors the suicide rate is only modestly predictable (CV R² ≈ 0.19) — suicide is multifactorial and national averages wash out much of the signal. The 90% conformal prediction intervals achieve ~96% empirical coverage (half-width ≈ 8.3 per 100k). We report every metric against a mean baseline, not a single lucky split.",
    followups: ["Why not a higher R²?", "What drives suicide risk?"],
  },
  {
    id: "leakage",
    keywords: ["leak", "leakage", "self-harm", "selfharm", "tautology", "circular", "0.77", "0.75", "why not"],
    answer:
      "The original model looked strong (R² ≈ 0.75) but that was mostly leakage: it used the IHME self-harm death rate to predict the WHO suicide rate, and self-harm ≈ suicide. Removing that circular feature drops cross-validated R² to ≈ 0.19 — the honest number. A cross-source check shows WHO and IHME agree strongly (Pearson r = 0.84), which is why using one to 'predict' the other was near-tautological.",
    followups: ["What drives suicide risk?", "How accurate is the model?"],
  },
  {
    id: "hierarchical",
    keywords: ["hierarch", "mixed", "icc", "region", "geographic", "cluster region", "pooling", "between region"],
    answer:
      "A hierarchical mixed-effects model (countries nested within WHO regions) gives an ICC of 0.22 — about 22% of the variance in national suicide rates is between regions. That's a clear geographic-clustering effect, and it's why flat models lean on region. Standardized fixed effects rank life expectancy (negative), health spending, unemployment, and alcohol (positive) as the strongest associations.",
    followups: ["What drives suicide risk?", "Show country clusters"],
  },
  {
    id: "subgroups",
    keywords: ["subgroup", "group", "high-risk", "high risk", "who is at risk", "profile", "african", "pattern"],
    answer:
      "Subgroup discovery (vs a 33% base rate of high-suicide countries) surfaces strong profiles: low-development countries in the African Region with low life expectancy reach a 76–80% share of high-suicide status. These are interpretable condition-combinations, not single variables.",
    followups: ["Show association rules", "What drives suicide risk?"],
  },
  {
    id: "rules",
    keywords: ["rule", "association", "fp-growth", "fpgrowth", "apriori", "lift", "combination", "co-occur"],
    answer:
      "FP-Growth association rules (with region/income/sex included) find non-trivial combinations, e.g. high addiction + high alcohol + low measured depression ⇒ high suicide (lift ≈ 2.1), and a distinct low-development African-region pattern. Lift > 1 means the combination is more associated with high suicide than chance.",
    followups: ["Show high-risk country groups", "Show country clusters"],
  },
  {
    id: "clusters",
    keywords: ["cluster", "umap", "network", "community", "graph", "similar", "embedding", "bridge"],
    answer:
      "Two unsupervised views on real data: a UMAP embedding groups countries into 5 profiles (with a clear suicide-rate gradient), and a k-nearest-neighbour similarity network finds 10 communities (modularity 0.72) with 'bridge' countries that connect otherwise separate clusters.",
    followups: ["Show high-risk country groups", "What drives suicide risk?"],
  },
  {
    id: "forecast",
    keywords: ["forecast", "time series", "n-beats", "nbeats", "deep learning", "predict future", "trend", "temporal", "lstm", "gru"],
    answer:
      "Forecasting is a methods demo on the synthetic panel (the only longitudinal data). Trained across all country series: a global N-BEATS (implemented in PyTorch and via darts) reached MAE ≈ 3.3–3.5, beating gradient-boosting (≈ 4.6–4.8) and a naive baseline (6.3). Because the data is synthetic, this shows the method works — not that real suicide rates are this forecastable.",
    followups: ["How accurate is the model?", "What are the versions?"],
  },
  {
    id: "versions",
    keywords: ["version", "v0", "v1", "v2", "v3", "dashboard", "difference", "streamlit", "layer"],
    answer:
      "Four versions: v0 is a static visual gallery (real data); v1 is the main dashboard + leakage-free ML (real data); v2 is an advanced-analytics methods showcase (synthetic); v3 is an interactive risk estimator with calibration and what-if scenarios (synthetic). Use the version cards above to open each dashboard.",
    followups: ["What drives suicide risk?", "Predict a country"],
  },
  {
    id: "predict_help",
    keywords: ["predict", "prediction", "estimate", "country", "rate for", "how to predict"],
    answer:
      "Type a country name (e.g. 'France' or 'Morocco') and I'll show its predicted age-standardized suicide rate with a 90% interval, plus the observed WHO value. You can also use the 'Predict a country' widget above. These are educational estimates, not clinical assessments.",
    followups: ["What drives suicide risk?", "How accurate is the model?"],
  },
  {
    id: "data",
    keywords: ["data", "source", "where from", "dataset", "who", "ihme", "world bank", "raw"],
    answer:
      "Sources are public: WHO suicide statistics (2021), IHME Global Burden of Disease (2023), and World Bank socioeconomic indicators — for 183 countries. I only share model results and predictions here, not the underlying datasets. The full data and code are in the GitHub repository.",
    followups: ["What drives suicide risk?", "What are the versions?"],
  },
  {
    id: "ethics",
    keywords: ["ethic", "clinical", "medical", "advice", "safe", "disclaimer", "help", "crisis", "diagnos"],
    answer:
      "This is an educational project on national, aggregate data — not a clinical tool, not medical advice, and not an individual risk assessment. If you or someone you know is struggling, please reach out to a local health professional or crisis line.",
    followups: ["What drives suicide risk?", "What are the versions?"],
  },
];

export const SUGGESTIONS = [
  "What drives suicide risk?",
  "How accurate is the model?",
  "Show high-risk country groups",
  "Predict a country",
  "What are the versions?",
];

export const FALLBACK =
  "I can help with the project's results — try one of these:";
