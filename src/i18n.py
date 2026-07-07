"""Lightweight EN/FR translation layer for the Streamlit app.

Covers the always-visible chrome (navigation, version labels, disclaimers, and the
v1 page names). Deeper in-page analytical captions fall back to English via `t()`
and can be translated incrementally by adding keys here.
"""
from __future__ import annotations

import streamlit as st

LANGS = ("en", "fr")

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "nav.title": "Navigation",
        "nav.goto": "Go to",
        "nav.switch": "Switch versions on the project website.",
        "lang.label": "Language / Langue",
        "guide.title": "Page guide",
        "guide.objective": "Objective",
        "guide.how": "How to use",
        "guide.notes": "Notes",
        "cg.objective": "Objective",
        "cg.snapshot": "Snapshot",
        "cg.how": "How to read",
        "cg.why": "Why this chart",
        "cg.current": "Current snapshot",
        "cg.default_snapshot": "Use the filters to update this view.",
        "ver.v0": "v0 · Visual gallery",
        "ver.v1": "v1 · Main dashboard",
        "ver.v2": "v2 · Advanced analytics",
        "ver.v3": "v3 · Risk estimator",
        "disc.real": (
            "**Educational use only — not clinical.** Built on public WHO (2021) & "
            "IHME GBD (2023) data for learning and storytelling. Not medical advice "
            "or diagnosis."
        ),
        "disc.synthetic": (
            "**Educational demo — not clinical.** v2/v3 run on **synthetic data** "
            "and illustrate methodology only. Outputs are **not** medical advice, "
            "diagnosis, or a risk assessment for any real person or country."
        ),
    },
    "fr": {
        "nav.title": "Navigation",
        "nav.goto": "Aller à",
        "nav.switch": "Changez de version sur le site du projet.",
        "lang.label": "Langue / Language",
        "guide.title": "Guide de la page",
        "guide.objective": "Objectif",
        "guide.how": "Utilisation",
        "guide.notes": "Notes",
        "cg.objective": "Objectif",
        "cg.snapshot": "Instantané",
        "cg.how": "Comment lire",
        "cg.why": "Pourquoi ce graphique",
        "cg.current": "Instantané actuel",
        "cg.default_snapshot": "Utilisez les filtres pour mettre à jour cette vue.",
        "ver.v0": "v0 · Galerie visuelle",
        "ver.v1": "v1 · Tableau de bord principal",
        "ver.v2": "v2 · Analytique avancée",
        "ver.v3": "v3 · Estimateur de risque",
        "disc.real": (
            "**Usage éducatif uniquement — non clinique.** Fondé sur des données "
            "publiques OMS (2021) & IHME GBD (2023) à des fins d'apprentissage. "
            "Ni avis médical, ni diagnostic."
        ),
        "disc.synthetic": (
            "**Démo éducative — non clinique.** v2/v3 utilisent des **données "
            "synthétiques** et illustrent la méthodologie uniquement. Les résultats "
            "ne sont **ni** un avis médical, **ni** un diagnostic, **ni** une "
            "évaluation du risque pour un pays ou une personne réelle."
        ),
    },
}

# v1 page names (English key -> French label). Unlisted names fall back to English.
PAGE_NAMES_FR: dict[str, str] = {
    "Overview / Story": "Aperçu / Récit",
    "WHO Suicide Explorer": "Explorateur suicide (OMS)",
    "Depression Burden (GBD)": "Fardeau de la dépression (GBD)",
    "Addiction (GBD)": "Addiction (GBD)",
    "Self-harm (GBD)": "Automutilation (GBD)",
    "Probability of Death (GBD)": "Probabilité de décès (GBD)",
    "All-cause Trends": "Tendances toutes causes",
    "Big Categories": "Grandes catégories",
    "Relationships": "Relations",
    "Country Segmentation": "Segmentation des pays",
    "Outliers & Alerts": "Valeurs aberrantes & alertes",
    "ML Demo": "Démo ML",
    "Methods + Data Model + Quality": "Méthodes + modèle de données + qualité",
    # v0 / v2 / v3 navigation
    "v0 Static Gallery": "v0 · Galerie statique",
    "v2 Overview": "v2 · Aperçu",
    "v2 Clusters": "v2 · Clusters",
    "v2 Trajectory Clusters": "v2 · Clusters de trajectoires",
    "v2 DTW Clusters": "v2 · Clusters DTW",
    "v2 Country Network": "v2 · Réseau de pays",
    "v2 Linked Views": "v2 · Vues liées",
    "v2 Forecasts": "v2 · Prévisions",
    "v2 Backtest": "v2 · Backtest",
    "v2 Scenario Lab": "v2 · Laboratoire de scénarios",
    "v2 Outliers": "v2 · Valeurs aberrantes",
    "v2 Patterns": "v2 · Motifs",
    "v3 Risk Estimator": "v3 · Estimateur de risque",
    "v3 Methods": "v3 · Méthodes",
}


# v1 section titles + subtitles (English source -> French). Missing keys fall back.
SECTIONS_FR: dict[str, str] = {
    "Overview": "Aperçu",
    "Story snapshot across WHO and GBD features.": "Instantané narratif des indicateurs OMS et GBD.",
    "WHO Suicide Explorer": "Explorateur suicide (OMS)",
    "Global and regional suicide patterns (2021).": "Schémas de suicide mondiaux et régionaux (2021).",
    "Depression Burden (DALYs)": "Fardeau de la dépression (DALYs)",
    "GBD DALYs rate for depressive disorders.": "Taux de DALYs (GBD) pour les troubles dépressifs.",
    "Addiction (Deaths Rate)": "Addiction (taux de décès)",
    "GBD substance-use mortality rates.": "Taux de mortalité liés aux substances (GBD).",
    "Self-harm (Deaths Rate)": "Automutilation (taux de décès)",
    "GBD self-harm mortality patterns.": "Schémas de mortalité par automutilation (GBD).",
    "Probability of Death": "Probabilité de décès",
    "Interpretation differs from rates: it is a probability, not a per-100k rate.":
        "L'interprétation diffère des taux : c'est une probabilité, pas un taux pour 100 000.",
    "All-cause Trends": "Tendances toutes causes",
    "DALYs trend across countries, WHO regions, and global aggregates.":
        "Tendance des DALYs par pays, régions OMS et agrégats mondiaux.",
    "Big Categories": "Grandes catégories",
    "GBD aggregate locations for big-category DALYs.":
        "Localisations agrégées GBD pour les DALYs par grande catégorie.",
    "Relationships": "Relations",
    "Correlation views from the merged ML dataset.":
        "Vues de corrélation à partir du jeu de données ML fusionné.",
    "Country Segmentation": "Segmentation des pays",
    "Unsupervised clustering of country profiles.": "Regroupement non supervisé des profils de pays.",
    "Outliers & Alerts": "Valeurs aberrantes & alertes",
    "Countries with unusual indicator patterns.": "Pays aux schémas d'indicateurs inhabituels.",
    "ML Demo": "Démo ML",
    "Ridge and RandomForest quick model comparison.":
        "Comparaison rapide des modèles Ridge et RandomForest.",
    "Methods, Data Model & Quality": "Méthodes, modèle de données & qualité",
    # --- v2 / v3 ---
    "v2 Synthetic Overview": "v2 · Aperçu synthétique",
    "Synthetic long-panel data for advanced demos.":
        "Données synthétiques en panel long pour démonstrations avancées.",
    "v2 Profile Clusters": "v2 · Clusters de profils",
    "KMeans clustering on synthetic 2023 profiles.":
        "Clustering KMeans sur les profils synthétiques 2023.",
    "v2 Trajectory Clusters": "v2 · Clusters de trajectoires",
    "Clusters based on 2000–2023 suicide-rate trajectories.":
        "Clusters basés sur les trajectoires de taux de suicide 2000–2023.",
    "v2 DTW Clusters": "v2 · Clusters DTW",
    "DTW clustering on 2000–2023 suicide-rate trajectories.":
        "Clustering DTW sur les trajectoires de taux de suicide 2000–2023.",
    "v2 Country Network": "v2 · Réseau de pays",
    "Similarity graph on 2023 profiles (cosine). Communities via modularity.":
        "Graphe de similarité sur les profils 2023 (cosinus). Communautés par modularité.",
    "v2 Linked Views": "v2 · Vues liées",
    "Brush the scatter to filter map + table. Small multiples for trends.":
        "Sélectionnez dans le nuage pour filtrer carte + table. Petits multiples pour les tendances.",
    "v2 Forecasts": "v2 · Prévisions",
    "Regional suicide-rate forecasts from synthetic data.":
        "Prévisions régionales du taux de suicide à partir de données synthétiques.",
    "v2 Backtest": "v2 · Backtest",
    "Rolling-origin backtest using lag features.":
        "Backtest à origine glissante utilisant des variables décalées.",
    "v2 Scenario Lab": "v2 · Laboratoire de scénarios",
    "What-if simulator using synthetic regression model.":
        "Simulateur hypothétique utilisant un modèle de régression synthétique.",
    "v2 Outliers": "v2 · Valeurs aberrantes",
    "IsolationForest anomalies (2023 profiles).": "Anomalies IsolationForest (profils 2023).",
    "v2 Patterns": "v2 · Motifs (règles)",
    "Association rules on binned 2023 profiles.":
        "Règles d'association sur les profils 2023 discrétisés.",
    "Methods & Synthetic Data": "Méthodes & données synthétiques",
    "v3 Risk Estimator": "v3 · Estimateur de risque",
    "Interactive probability of high-risk suicide category.":
        "Probabilité interactive de catégorie de suicide à haut risque.",
    "v3 Methods": "v3 · Méthodes",
    "Model scope, inputs, and dataset notes.":
        "Portée du modèle, entrées et notes sur le jeu de données.",
    # --- in-page markdown sub-headers (### / ####) ---
    "### Analytics notes": "### Notes analytiques",
    "### Backtest metrics": "### Métriques de backtest",
    "### By age group": "### Par groupe d'âge",
    "### Calibration reliability": "### Fiabilité de la calibration",
    "### Cluster centers (original units)": "### Centres des clusters (unités d'origine)",
    "### Cluster centers (trajectory features)": "### Centres des clusters (variables de trajectoire)",
    "### Community sizes": "### Tailles des communautés",
    "### Correlation heatmap": "### Carte de chaleur des corrélations",
    "### Counterfactual hints (10% reduction)": "### Indices contrefactuels (réduction de 10 %)",
    "### Cross-validation (5-fold)": "### Validation croisée (5 plis)",
    "### Crude vs age-standardized": "### Brut vs standardisé par âge",
    "### DL forecast metrics": "### Métriques de prévision (DL)",
    "### Data Model (Star Schema)": "### Modèle de données (schéma en étoile)",
    "### Data Quality (Great Expectations)": "### Qualité des données (Great Expectations)",
    "### Data Quality Scorecard": "### Fiche de qualité des données",
    "### Explainability": "### Explicabilité",
    "### Feature importance (RandomForest)": "### Importance des variables (RandomForest)",
    "### K selection (silhouette)": "### Sélection de k (silhouette)",
    "### K selection (silhouette/inertia)": "### Sélection de k (silhouette/inertie)",
    "### KPI Benchmarking (2021 global percentiles)": "### Benchmark des indicateurs (percentiles mondiaux 2021)",
    "### Local feature contributions": "### Contributions locales des variables",
    "### Methods breakdown": "### Répartition des méthodes",
    "### Model demo signal (quick check)": "### Signal de démo du modèle (vérification rapide)",
    "### Model diagnostics (training set)": "### Diagnostics du modèle (jeu d'entraînement)",
    "### Model metrics": "### Métriques du modèle",
    "### Prediction intervals (quantile regression)": "### Intervalles de prédiction (régression quantile)",
    "### Quantile model metrics": "### Métriques du modèle quantile",
    "### Rule interpretation": "### Interprétation des règles",
    "### Sensitivity (10% increase per feature)": "### Sensibilité (hausse de 10 % par variable)",
    "### Sex comparison": "### Comparaison par sexe",
    "### Similar countries (nearest neighbors)": "### Pays similaires (plus proches voisins)",
    "### Small multiples (regional trends)": "### Petits multiples (tendances régionales)",
    "### Strongest edges": "### Liens les plus forts",
    "### Synthetic generation notes": "### Notes de génération synthétique",
    "### Synthetic validity report": "### Rapport de validité synthétique",
    "### Top anomalies": "### Principales anomalies",
    "### Top central countries": "### Pays les plus centraux",
    "### Top rules (by lift)": "### Meilleures règles (par lift)",
    "#### DL forecast": "#### Prévision DL",
    "#### DTW clustering": "#### Clustering DTW",
    "#### Graph clustering": "#### Clustering de graphe",
    "#### ISO3 unmatched (by source_type)": "#### ISO3 non appariés (par source_type)",
    "#### Trajectory clustering": "#### Clustering de trajectoires",
}


def section(text: str) -> str:
    """Translate a v1 section title/subtitle string (falls back to the English)."""
    if current_lang() == "fr":
        return SECTIONS_FR.get(text, text)
    return text


def current_lang() -> str:
    q = st.query_params.get("lang")
    if q in LANGS:
        st.session_state["mhv_lang"] = q
    return st.session_state.get("mhv_lang", "en")


def t(key: str) -> str:
    lang = current_lang()
    return TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS["en"].get(key, key)


def tr(en: str) -> str:
    """Translate a widget label / caption / message (falls back to English)."""
    if current_lang() == "fr":
        from i18n_fr import UI_FR

        return UI_FR.get(en, en)
    return en


def fmt(en_template: str, *args) -> str:
    """Translate a template with {} placeholders, then format with args.

    Example: fmt("Selection: {} countries.", n).
    """
    tmpl = en_template
    if current_lang() == "fr":
        from i18n_fr import FMT_FR

        tmpl = FMT_FR.get(en_template, en_template)
    return tmpl.format(*args)


def label_value(value, category: str):
    """Translate a *data value* shown in a dropdown for display only.

    The raw ``value`` is what gets returned/filtered on; this only changes the
    label the user sees. Use as ``format_func=lambda v: i18n.label_value(v, "sex")``.
    Categories: "sex", "age", "cause", "region", "metric", "location_type".
    Unknown values fall back to the original string unchanged.
    """
    if current_lang() != "fr":
        return value
    from i18n_fr import VALUE_FR

    return VALUE_FR.get(category, {}).get(value, value)


def vf(category: str):
    """Return a ``format_func`` callable for a data-value ``category``."""
    return lambda v: label_value(v, category)


def page_label(name: str) -> str:
    if current_lang() == "fr":
        return PAGE_NAMES_FR.get(name, name)
    return name


def language_selector() -> str:
    lang = current_lang()
    choice = st.sidebar.radio(
        t("lang.label"),
        list(LANGS),
        index=list(LANGS).index(lang),
        format_func=lambda x: {"en": "English", "fr": "Français"}[x],
        horizontal=True,
    )
    st.session_state["mhv_lang"] = choice
    if st.query_params.get("lang") != choice:
        st.query_params["lang"] = choice
    return choice
