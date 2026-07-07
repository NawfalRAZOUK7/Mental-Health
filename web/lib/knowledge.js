// Bilingual (EN/FR) knowledge base for the guide chatbot. Answers come from the
// project's RESULTS only (ML / DM / model predictions) — no raw data, no LLM.
// Keyword matching checks both languages, so users can type in EN or FR.

// Short chip labels for every id that can appear as a suggestion or follow-up.
export const CHIP_LABELS = {
  drivers: { en: "What drives suicide risk?", fr: "Quels sont les facteurs de risque ?" },
  accuracy: { en: "How accurate is the model?", fr: "Quelle est la précision du modèle ?" },
  leakage: { en: "Was the model leaking?", fr: "Y avait-il une fuite de données ?" },
  subgroups: { en: "High-risk country groups", fr: "Groupes de pays à haut risque" },
  rules: { en: "Association rules", fr: "Règles d'association" },
  clusters: { en: "Clusters & network", fr: "Clusters & réseau" },
  forecast: { en: "Forecasting", fr: "Prévision" },
  versions: { en: "What are the versions?", fr: "Quelles sont les versions ?" },
  predict: { en: "Predict a country", fr: "Prédire un pays" },
  data: { en: "Data sources", fr: "Sources de données" },
  ethics: { en: "Is this clinical?", fr: "Est-ce clinique ?" },
};

// The chips shown initially and on fallback.
export const SUGGESTIONS = ["drivers", "accuracy", "subgroups", "predict", "versions"];

export const CHAT = {
  en: {
    open: "Ask the guide",
    close: "Close guide",
    header: "Project guide",
    placeholder: "Ask about results, or type a country…",
    send: "Send",
    note: "Educational guide · answers from project results, not raw data · no medical advice",
    greeting: "Hi! I'm the project guide. Ask me about the findings, the models, or type a country to see its prediction. I only share results — not the raw data.",
    fallback: "I can help with the project's results — try one of these:",
    country: (p) => `${p.name}: predicted ${p.pred.toFixed(1)} per 100k (90% interval ${p.lower.toFixed(1)}–${p.upper.toFixed(1)}); observed WHO rate ${p.actual.toFixed(1)}. Educational estimate, not a clinical assessment.`,
  },
  fr: {
    open: "Demander au guide",
    close: "Fermer",
    header: "Guide du projet",
    placeholder: "Posez une question sur les résultats, ou tapez un pays…",
    send: "Envoyer",
    note: "Guide éducatif · réponses issues des résultats du projet, pas des données brutes · pas d'avis médical",
    greeting: "Bonjour ! Je suis le guide du projet. Posez-moi des questions sur les résultats, les modèles, ou tapez un pays pour voir sa prédiction. Je partage uniquement les résultats — pas les données brutes.",
    fallback: "Je peux vous aider avec les résultats du projet — essayez ceci :",
    country: (p) => `${p.name} : taux prédit ${p.pred.toFixed(1)} pour 100 000 (intervalle 90 % ${p.lower.toFixed(1)}–${p.upper.toFixed(1)}) ; taux observé OMS ${p.actual.toFixed(1)}. Estimation éducative, pas une évaluation clinique.`,
  },
};

export const INTENTS = [
  {
    id: "drivers",
    keywords: {
      en: ["driver", "drive", "cause", "factor", "risk", "shap", "important", "influence", "why", "raise", "increase"],
      fr: ["facteur", "cause", "risque", "important", "influence", "pourquoi", "augmente", "moteur", "explique"],
    },
    answer: {
      en: "The strongest independent drivers of the national suicide rate (from SHAP on the leakage-free model) are life expectancy and alcohol consumption per capita, followed by addiction burden, GDP per capita, and urbanization. Lower life expectancy and higher alcohol use push predicted risk up.",
      fr: "Les principaux facteurs indépendants du taux de suicide national (d'après SHAP sur le modèle sans fuite) sont l'espérance de vie et la consommation d'alcool par habitant, suivies du fardeau de l'addiction, du PIB par habitant et de l'urbanisation. Une espérance de vie plus faible et une consommation d'alcool plus élevée augmentent le risque prédit.",
    },
    followups: ["accuracy", "subgroups"],
  },
  {
    id: "accuracy",
    keywords: {
      en: ["accurate", "accuracy", "r2", "mae", "error", "performance", "reliable", "score", "metric", "how well"],
      fr: ["précision", "précis", "fiable", "erreur", "performance", "score", "métrique", "exact"],
    },
    answer: {
      en: "Honest, cross-validated numbers: on independent predictors the suicide rate is only modestly predictable (CV R² ≈ 0.19) — suicide is multifactorial and national averages wash out much of the signal. The 90% conformal intervals achieve ~96% empirical coverage. Every metric is reported against a mean baseline.",
      fr: "Des chiffres honnêtes, en validation croisée : sur des prédicteurs indépendants, le taux de suicide n'est que modérément prévisible (R² CV ≈ 0,19) — le suicide est multifactoriel et les moyennes nationales atténuent le signal. Les intervalles conformes à 90 % atteignent une couverture empirique d'environ 96 %. Chaque métrique est comparée à une référence moyenne.",
    },
    followups: ["drivers", "subgroups"],
  },
  {
    id: "leakage",
    keywords: {
      en: ["leak", "leakage", "self-harm", "selfharm", "circular", "0.77", "0.75", "tautology"],
      fr: ["fuite", "automutilation", "circulaire", "tautologie", "biais"],
    },
    answer: {
      en: "The original model looked strong (R² ≈ 0.75) but that was mostly leakage: it predicted the WHO suicide rate from the IHME self-harm rate, and self-harm ≈ suicide. Removing that circular feature drops cross-validated R² to ≈ 0.19 — the honest number. WHO and IHME agree strongly (r = 0.84), which is why using one to predict the other was near-tautological.",
      fr: "Le modèle initial semblait performant (R² ≈ 0,75) mais il s'agissait surtout d'une fuite de données : il prédisait le taux de suicide de l'OMS à partir du taux d'automutilation de l'IHME, or automutilation ≈ suicide. En retirant cette variable circulaire, le R² en validation croisée tombe à ≈ 0,19 — le chiffre honnête. L'OMS et l'IHME concordent fortement (r = 0,84), d'où le caractère quasi tautologique.",
    },
    followups: ["drivers", "accuracy"],
  },
  {
    id: "subgroups",
    keywords: {
      en: ["subgroup", "group", "high-risk", "high risk", "profile", "african", "pattern", "who is at risk"],
      fr: ["sous-groupe", "groupe", "haut risque", "profil", "africain", "motif", "qui est à risque"],
    },
    answer: {
      en: "Subgroup discovery (vs a 33% base rate of high-suicide countries) surfaces strong profiles: low-development countries in the African Region with low life expectancy reach a 76–80% share of high-suicide status. These are interpretable condition-combinations, not single variables.",
      fr: "La découverte de sous-groupes (contre un taux de base de 33 % de pays à suicide élevé) révèle des profils marqués : les pays peu développés de la Région africaine à faible espérance de vie atteignent 76–80 % de statut « suicide élevé ». Ce sont des combinaisons de conditions interprétables, pas des variables isolées.",
    },
    followups: ["rules", "drivers"],
  },
  {
    id: "rules",
    keywords: {
      en: ["rule", "association", "fp-growth", "apriori", "lift", "combination", "co-occur"],
      fr: ["règle", "association", "lift", "combinaison", "cooccurrence"],
    },
    answer: {
      en: "FP-Growth association rules (with region/income/sex included) find non-trivial combinations, e.g. high addiction + high alcohol + low measured depression ⇒ high suicide (lift ≈ 2.1). Lift > 1 means the combination is more associated with high suicide than chance.",
      fr: "Les règles d'association FP-Growth (incluant région/revenu/sexe) trouvent des combinaisons non triviales, p. ex. forte addiction + forte consommation d'alcool + faible dépression mesurée ⇒ suicide élevé (lift ≈ 2,1). Un lift > 1 signifie que la combinaison est plus associée au suicide élevé que le hasard.",
    },
    followups: ["subgroups", "clusters"],
  },
  {
    id: "clusters",
    keywords: {
      en: ["cluster", "umap", "network", "community", "graph", "similar", "embedding", "bridge"],
      fr: ["cluster", "umap", "réseau", "communauté", "graphe", "similaire", "plongement", "pont", "regroupement"],
    },
    answer: {
      en: "Two unsupervised views on real data: a UMAP embedding groups countries into 5 profiles (with a clear suicide-rate gradient), and a k-nearest-neighbour similarity network finds 10 communities (modularity 0.72) with 'bridge' countries connecting otherwise separate clusters.",
      fr: "Deux vues non supervisées sur données réelles : un plongement UMAP regroupe les pays en 5 profils (avec un gradient net de taux de suicide), et un réseau de similarité (k plus proches voisins) trouve 10 communautés (modularité 0,72) avec des pays « ponts » reliant des clusters distincts.",
    },
    followups: ["subgroups", "drivers"],
  },
  {
    id: "forecast",
    keywords: {
      en: ["forecast", "time series", "n-beats", "deep learning", "future", "trend", "temporal", "lstm", "gru"],
      fr: ["prévision", "prédire l'avenir", "série temporelle", "apprentissage profond", "tendance", "temporel"],
    },
    answer: {
      en: "Forecasting is a methods demo on the synthetic panel (the only longitudinal data). Trained across all country series, a global N-BEATS (PyTorch and darts) reached MAE ≈ 3.3–3.5, beating gradient-boosting (≈ 4.6–4.8) and a naive baseline (6.3). Because the data is synthetic, this shows the method works — not that real rates are this forecastable.",
      fr: "La prévision est une démonstration de méthode sur le panel synthétique (seules données longitudinales). Entraîné sur toutes les séries pays, un N-BEATS global (PyTorch et darts) atteint une MAE ≈ 3,3–3,5, battant le gradient boosting (≈ 4,6–4,8) et une référence naïve (6,3). Les données étant synthétiques, cela montre que la méthode fonctionne — pas que les taux réels sont aussi prévisibles.",
    },
    followups: ["accuracy", "versions"],
  },
  {
    id: "versions",
    keywords: {
      en: ["version", "v0", "v1", "v2", "v3", "dashboard", "difference", "streamlit"],
      fr: ["version", "v0", "v1", "v2", "v3", "tableau de bord", "différence"],
    },
    answer: {
      en: "Four versions: v0 is a static visual gallery (real data); v1 is the main dashboard + leakage-free ML (real data); v2 is an advanced-analytics methods showcase (synthetic); v3 is an interactive risk estimator with calibration and what-if scenarios (synthetic). Use the version cards to open each dashboard.",
      fr: "Quatre versions : v0 est une galerie visuelle statique (données réelles) ; v1 est le tableau de bord principal + ML sans fuite (données réelles) ; v2 est une vitrine de méthodes avancées (synthétique) ; v3 est un estimateur de risque interactif avec calibration et scénarios (synthétique). Utilisez les cartes pour ouvrir chaque tableau de bord.",
    },
    followups: ["drivers", "predict"],
  },
  {
    id: "predict",
    keywords: {
      en: ["predict", "prediction", "estimate", "country", "rate for"],
      fr: ["prédire", "prédiction", "estimer", "pays", "taux pour"],
    },
    answer: {
      en: "Type a country name (e.g. 'France' or 'Morocco') and I'll show its predicted suicide rate with a 90% interval, plus the observed WHO value. You can also use the 'Predict a country' widget above. Educational estimates, not clinical assessments.",
      fr: "Tapez le nom d'un pays (p. ex. « France » ou « Morocco ») et j'afficherai son taux de suicide prédit avec un intervalle à 90 %, ainsi que la valeur observée de l'OMS. Vous pouvez aussi utiliser le widget « Prédire un pays » ci-dessus. Estimations éducatives, pas cliniques.",
    },
    followups: ["drivers", "accuracy"],
  },
  {
    id: "data",
    keywords: {
      en: ["data", "source", "dataset", "who", "ihme", "world bank", "raw"],
      fr: ["données", "source", "jeu de données", "oms", "ihme", "banque mondiale", "brutes"],
    },
    answer: {
      en: "Sources are public: WHO suicide statistics (2021), IHME Global Burden of Disease (2023), and World Bank indicators — for 183 countries. I only share model results and predictions here, not the underlying datasets. The full data and code are in the GitHub repository.",
      fr: "Les sources sont publiques : statistiques de l'OMS sur le suicide (2021), IHME Global Burden of Disease (2023) et indicateurs de la Banque mondiale — pour 183 pays. Je ne partage ici que les résultats et prédictions du modèle, pas les jeux de données sous-jacents. Les données et le code complets sont dans le dépôt GitHub.",
    },
    followups: ["drivers", "versions"],
  },
  {
    id: "ethics",
    keywords: {
      en: ["ethic", "clinical", "medical", "advice", "safe", "disclaimer", "help", "crisis"],
      fr: ["éthique", "clinique", "médical", "conseil", "avertissement", "aide", "crise"],
    },
    answer: {
      en: "This is an educational project on national, aggregate data — not a clinical tool, not medical advice, and not an individual risk assessment. If you or someone you know is struggling, please reach out to a local health professional or crisis line.",
      fr: "Ceci est un projet éducatif sur des données nationales et agrégées — pas un outil clinique, pas un avis médical, et pas une évaluation individuelle du risque. Si vous ou une personne de votre entourage traversez une période difficile, contactez un professionnel de santé local ou une ligne d'écoute.",
    },
    followups: ["drivers", "versions"],
  },
];
