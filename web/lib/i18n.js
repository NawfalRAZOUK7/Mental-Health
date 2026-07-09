// EN/FR UI strings for the website. Access with t("key") via the language hook.
// Chatbot content lives (bilingual) in knowledge.js.
export const LANGS = ["en", "fr"];
export const DEFAULT_LANG = "en";

export const translations = {
  en: {
    "nav.try": "Try the predictor",
    "nav.explore": "Explore the dashboards",
    "nav.code": "Code on GitHub",

    "disclaimer": "Educational project — not a clinical tool, not medical advice. National, aggregate estimates only.",

    "hero.eyebrow": "WHO 2021 · IHME GBD 2023 · World Bank",
    "hero.h1": "Turning global mental-health data into clear, honest insight.",
    "hero.lede": "A full-stack analytics project: reproducible pipelines, leakage-free machine learning, data mining, deep-learning forecasting, and a live predictor of national suicide rates — built for transparency, not hype.",

    "findings.title": "What the data actually says",
    "findings.sub": "Findings from the real WHO + IHME + World Bank data (183 countries).",
    "findings.s1.v": "r = 0.84",
    "findings.s1.l": "WHO and IHME agree on measuring suicide / self-harm (Pearson)",
    "findings.s2.v": "0.75 → 0.19",
    "findings.s2.l": "Cross-validated R² drops once the circular self-harm feature is removed — most apparent skill was leakage",
    "findings.s3.v": "Life expectancy & alcohol",
    "findings.s3.l": "Strongest independent drivers of the national suicide rate (SHAP)",

    "predict.title": "Predict a country",
    "predict.sub": "A leakage-free model estimates the age-standardized suicide rate (per 100k) with a 90% conformal interval. Runs entirely in your browser — predictions are baked in from the trained model.",
    "predictor.choose": "Choose a country",
    "predictor.select": "— select —",
    "predictor.predicted": "predicted",
    "predictor.per100k": "per 100k",
    "predictor.interval": "90% interval:",
    "predictor.observed": "observed (WHO):",
    "predictor.legPred": "model prediction",
    "predictor.legObs": "observed rate",
    "predictor.legInt": "90% interval",
    "predictor.scale": "scale",
    "predictor.note": "Educational estimate, not a clinical or individual risk assessment.",

    "versions.title": "Explore every version",
    "versions.sub": "Four progressive dashboards, from static visuals to an interactive risk estimator. Each opens in a new tab.",
    "versions.open": "Open dashboard →",
    "versions.tagReal": "Real data",
    "versions.tagSynthetic": "Synthetic",
    "versions.v0.title": "v0 · Visual gallery",
    "versions.v0.blurb": "Static, high-variety visuals straight from the raw WHO & IHME data.",
    "versions.v1.title": "v1 · Main dashboard",
    "versions.v1.blurb": "Real-data BI dashboard, ML baseline, and the leakage-free enriched model.",
    "versions.v2.title": "v2 · Advanced analytics",
    "versions.v2.blurb": "Methods showcase: clustering, forecasting, graphs, explainability.",
    "versions.v3.title": "v3 · Risk estimator",
    "versions.v3.blurb": "Interactive probability tool with calibration and what-if scenarios.",

    "methods.title": "Under the hood",
    "methods.sub": "Modern, honest methods — with baselines kept visible so nothing is oversold.",
    "fig.shap": "SHAP — low life expectancy and high alcohol consumption push predicted risk up.",
    "fig.umap": "UMAP — countries mapped by mental-health profile, with a suicide-rate gradient.",
    "fig.network": "Similarity network — 10 communities (modularity 0.72) with bridge countries.",
    "fig.agreement": "Cross-source check — WHO vs IHME agreement and measurement difference.",

    "built.title": "How it's built",
    "built.ml": "Machine learning",
    "built.mlItems": ["LightGBM + Optuna, nested CV", "SHAP explainability", "Conformal prediction intervals", "Hierarchical mixed-effects (ICC)"],
    "built.dm": "Data mining",
    "built.dmItems": ["UMAP embedding", "Country similarity network", "Subgroup discovery", "FP-Growth association rules"],
    "built.eng": "Engineering",
    "built.engItems": ["Reproducible pipelines", "FastAPI prediction service", "Docker + docker-compose", "CI, tests, LaTeX report"],

    "footer.p1": "Mental Health Viz — educational analytics on public data. Sources: WHO suicide statistics (2021), IHME Global Burden of Disease (2023), World Bank Open Data.",
    "footer.p2": "Not medical advice. If you or someone you know is struggling, please contact a local health professional or crisis line. Code: MIT · Content: CC BY 4.0.",
    "footer.github": "GitHub repository",
    "footer.author": "By Nawfal RAZOUK",
    "footer.supervisor": "Supervised by Nabila ZRIRA (ENSMR)",
  },

  fr: {
    "nav.try": "Essayer le prédicteur",
    "nav.explore": "Explorer les tableaux de bord",
    "nav.code": "Code sur GitHub",

    "disclaimer": "Projet éducatif — ce n'est pas un outil clinique ni un avis médical. Estimations nationales et agrégées uniquement.",

    "hero.eyebrow": "OMS 2021 · IHME GBD 2023 · Banque mondiale",
    "hero.h1": "Transformer les données mondiales de santé mentale en analyses claires et honnêtes.",
    "hero.lede": "Un projet d'analyse complet : pipelines reproductibles, apprentissage automatique sans fuite de données, fouille de données, prévision par apprentissage profond et un prédicteur en direct des taux de suicide nationaux — conçu pour la transparence, pas le sensationnel.",

    "findings.title": "Ce que disent réellement les données",
    "findings.sub": "Résultats issus des données réelles OMS + IHME + Banque mondiale (183 pays).",
    "findings.s1.v": "r = 0,84",
    "findings.s1.l": "L'OMS et l'IHME concordent sur la mesure du suicide / de l'automutilation (Pearson)",
    "findings.s2.v": "0,75 → 0,19",
    "findings.s2.l": "Le R² en validation croisée chute une fois retirée la variable circulaire d'automutilation — l'essentiel de la performance apparente était une fuite de données",
    "findings.s3.v": "Espérance de vie & alcool",
    "findings.s3.l": "Principaux facteurs indépendants du taux de suicide national (SHAP)",

    "predict.title": "Prédire un pays",
    "predict.sub": "Un modèle sans fuite de données estime le taux de suicide standardisé par âge (pour 100 000) avec un intervalle conforme à 90 %. Tout s'exécute dans votre navigateur — les prédictions sont intégrées à partir du modèle entraîné.",
    "predictor.choose": "Choisir un pays",
    "predictor.select": "— sélectionner —",
    "predictor.predicted": "prédit",
    "predictor.per100k": "pour 100 000",
    "predictor.interval": "intervalle 90 % :",
    "predictor.observed": "observé (OMS) :",
    "predictor.legPred": "prédiction du modèle",
    "predictor.legObs": "taux observé",
    "predictor.legInt": "intervalle 90 %",
    "predictor.scale": "échelle",
    "predictor.note": "Estimation éducative, pas une évaluation clinique ou individuelle du risque.",

    "versions.title": "Explorer chaque version",
    "versions.sub": "Quatre tableaux de bord progressifs, des visuels statiques à un estimateur de risque interactif. Chacun s'ouvre dans un nouvel onglet.",
    "versions.open": "Ouvrir le tableau de bord →",
    "versions.tagReal": "Données réelles",
    "versions.tagSynthetic": "Synthétique",
    "versions.v0.title": "v0 · Galerie visuelle",
    "versions.v0.blurb": "Visuels statiques et variés, directement issus des données brutes OMS & IHME.",
    "versions.v1.title": "v1 · Tableau de bord principal",
    "versions.v1.blurb": "Tableau de bord sur données réelles, référence ML et modèle enrichi sans fuite.",
    "versions.v2.title": "v2 · Analytique avancée",
    "versions.v2.blurb": "Vitrine de méthodes : clustering, prévision, graphes, explicabilité.",
    "versions.v3.title": "v3 · Estimateur de risque",
    "versions.v3.blurb": "Outil de probabilité interactif avec calibration et scénarios hypothétiques.",

    "methods.title": "Sous le capot",
    "methods.sub": "Des méthodes modernes et honnêtes — les références restent visibles pour ne rien survendre.",
    "fig.shap": "SHAP — une faible espérance de vie et une forte consommation d'alcool augmentent le risque prédit.",
    "fig.umap": "UMAP — pays cartographiés par profil de santé mentale, avec un gradient de taux de suicide.",
    "fig.network": "Réseau de similarité — 10 communautés (modularité 0,72) avec des pays « ponts ».",
    "fig.agreement": "Vérification inter-sources — concordance OMS vs IHME et écart de mesure.",

    "built.title": "Comment c'est construit",
    "built.ml": "Apprentissage automatique",
    "built.mlItems": ["LightGBM + Optuna, CV imbriquée", "Explicabilité SHAP", "Intervalles de prédiction conformes", "Modèle hiérarchique à effets mixtes (ICC)"],
    "built.dm": "Fouille de données",
    "built.dmItems": ["Plongement UMAP", "Réseau de similarité des pays", "Découverte de sous-groupes", "Règles d'association FP-Growth"],
    "built.eng": "Ingénierie",
    "built.engItems": ["Pipelines reproductibles", "Service de prédiction FastAPI", "Docker + docker-compose", "CI, tests, rapport LaTeX"],

    "footer.p1": "Mental Health Viz — analyses éducatives sur données publiques. Sources : statistiques de l'OMS sur le suicide (2021), IHME Global Burden of Disease (2023), World Bank Open Data.",
    "footer.p2": "Ceci n'est pas un avis médical. Si vous ou une personne de votre entourage traversez une période difficile, contactez un professionnel de santé local ou une ligne d'écoute. Code : MIT · Contenu : CC BY 4.0.",
    "footer.github": "Dépôt GitHub",
    "footer.author": "Par Nawfal RAZOUK",
    "footer.supervisor": "Encadré par Nabila ZRIRA (ENSMR)",
  },
};

export function makeT(lang) {
  const dict = translations[lang] || translations.en;
  return (key) => (key in dict ? dict[key] : (translations.en[key] ?? key));
}
