"""French translations for the Streamlit app's widget labels, captions, and messages.

Pure data (no imports). Used by i18n.tr(). Keyed by the exact English source string.
f-string messages with runtime values are handled separately in app.py.
CHART_GUIDES_FR (bottom) holds French chart-guide entries keyed by chart_key.
"""

UI_FR: dict[str, str] = {
    # --- Widget labels ---
    "Country": "Pays",
    "Sex": "Sexe",
    "Sex (intervals)": "Sexe (intervalles)",
    "Sex (multiples)": "Sexe (multiples)",
    "Year": "Année",
    "Metric": "Mesure",
    "Metric (multiples)": "Mesure (multiples)",
    "Region": "Région",
    "Region (intervals)": "Région (intervalles)",
    "Cause": "Cause",
    "Cluster": "Cluster",
    "Age group": "Groupe d'âge",
    "Aggregate location": "Localisation agrégée",
    "Benchmark metric": "Métrique de référence",
    "Chart type": "Type de graphique",
    "Chart types": "Types de graphiques",
    "Location": "Lieu",
    "Location type": "Type de lieu",
    "Forecast model": "Modèle de prévision",
    "Test size": "Taille du test",
    "Trend view": "Vue des tendances",
    "Training data source": "Source des données d'entraînement",
    "High-risk cutoff percentile": "Percentile seuil de haut risque",
    "Depression DALYs rate": "Taux de DALYs de dépression",
    "Addiction death rate": "Taux de décès par addiction",
    "Self-harm death rate": "Taux de décès par automutilation",
    "Enable brush selection": "Activer la sélection par balayage",
    "Print / PDF mode": "Mode Impression / PDF",
    "Interactive version": "Version interactive",
    "Show HTML": "Afficher le HTML",
    "Explain chart": "Expliquer le graphique",
    "How to read this page": "Comment lire cette page",
    "Datasets": "Jeux de données",
    "Data Dictionary": "Dictionnaire de données",
    "Synthetic data dictionary": "Dictionnaire de données synthétiques",
    "Advanced methods notes": "Notes de méthodes avancées",
    "Open Great Expectations HTML report": "Ouvrir le rapport HTML Great Expectations",
    "Target": "Cible",
    "X axis": "Axe X",
    "Y axis": "Axe Y",
    "X variable": "Variable X",
    "Y variable": "Variable Y",
    # --- Captions / info / error messages ---
    "Backtest predictions are empty. Re-run src/v2_backtest.py to regenerate.":
        "Les prédictions de backtest sont vides. Relancez src/v2_backtest.py pour les régénérer.",
    "Bands show q10–q90 intervals; q50 is the median prediction.":
        "Les bandes montrent les intervalles q10–q90 ; q50 est la prédiction médiane.",
    "Calibration: isotonic (cv=3).": "Calibration : isotonique (cv=3).",
    "Calibration: off (insufficient data or sklearn unavailable).":
        "Calibration : désactivée (données insuffisantes ou sklearn indisponible).",
    "Closer to the diagonal means probabilities match observed rates.":
        "Plus c'est proche de la diagonale, mieux les probabilités correspondent aux taux observés.",
    "Closest countries in feature space (standardized numeric inputs).":
        "Pays les plus proches dans l'espace des variables (entrées numériques standardisées).",
    "Data quality scorecard not found. Run src/07_data_quality_scorecard.py.":
        "Fiche de qualité des données introuvable. Lancez src/07_data_quality_scorecard.py.",
    "Each line is a country. The region trend is the overall band of country trajectories.":
        "Chaque ligne est un pays. La tendance régionale est la bande globale des trajectoires nationales.",
    "Honest generalization estimate from held-out folds. ":
        "Estimation honnête de généralisation à partir des plis retenus. ",
    "In-sample training metrics are shown afterward and are optimistic.":
        "Les métriques d'entraînement (sur échantillon) sont affichées ensuite et sont optimistes.",
    "In-sample only (too few positives for cross-validation); optimistic.":
        "Sur échantillon d'entraînement uniquement (trop peu de positifs pour la validation croisée) ; optimiste.",
    "Inputs are bounded to the 10th–90th percentile of recent synthetic data.":
        "Les entrées sont bornées aux 10e–90e percentiles des données synthétiques récentes.",
    "Install scikit-learn to enable the v3 estimator.":
        "Installez scikit-learn pour activer l'estimateur v3.",
    "Line = regional aggregate; shaded band = country interquartile range (25th–75th).":
        "Ligne = agrégat régional ; bande ombrée = intervalle interquartile des pays (25e–75e).",
    "Methods breakdown not available in the filtered self-harm dataset.":
        "Répartition des méthodes indisponible dans le jeu de données automutilation filtré.",
    "Missing v0/assets/manifest.csv. Generate it to power the gallery.":
        "v0/assets/manifest.csv manquant. Générez-le pour alimenter la galerie.",
    "No baseline data available for Scenario Lab.":
        "Aucune donnée de référence disponible pour le Laboratoire de scénarios.",
    "No charts match the current filters.": "Aucun graphique ne correspond aux filtres actuels.",
    "No data available for this selection. Try another year or sex.":
        "Aucune donnée disponible pour cette sélection. Essayez une autre année ou un autre sexe.",
    "No outliers detected with the current threshold.":
        "Aucune valeur aberrante détectée avec le seuil actuel.",
    "No quantile predictions for this selection.":
        "Aucune prédiction quantile pour cette sélection.",
    "Not causal; shows local sensitivity if one feature is reduced by 10%.":
        "Non causal ; montre la sensibilité locale si une variable est réduite de 10 %.",
    "Not enough data for ML demo.": "Pas assez de données pour la démo ML.",
    "Print mode enabled. Use browser Print → Save as PDF for submission.":
        "Mode impression activé. Utilisez Imprimer du navigateur → Enregistrer en PDF.",
    "Region aggregates not available. Showing country trajectories.":
        "Agrégats régionaux indisponibles. Affichage des trajectoires nationales.",
    "Run python scripts/run_v3_pipeline.py to generate v3 feature tables.":
        "Lancez python scripts/run_v3_pipeline.py pour générer les tables de variables v3.",
    "Run src/06_ml_baseline.py to generate cross-validation metrics.":
        "Lancez src/06_ml_baseline.py pour générer les métriques de validation croisée.",
    "Run src/08_segmentation_outliers.py to generate outlier outputs.":
        "Lancez src/08_segmentation_outliers.py pour générer les sorties de valeurs aberrantes.",
    "Run src/08_segmentation_outliers.py to generate segmentation outputs.":
        "Lancez src/08_segmentation_outliers.py pour générer les sorties de segmentation.",
    "Run src/v2_analytics.py to generate cluster outputs.":
        "Lancez src/v2_analytics.py pour générer les sorties de clusters.",
    "Run src/v2_analytics.py to generate forecast outputs.":
        "Lancez src/v2_analytics.py pour générer les sorties de prévision.",
    "Run src/v2_analytics.py to generate model coefficients.":
        "Lancez src/v2_analytics.py pour générer les coefficients du modèle.",
    "Run src/v2_analytics.py to generate outlier outputs.":
        "Lancez src/v2_analytics.py pour générer les sorties de valeurs aberrantes.",
    "Run src/v2_assoc_rules.py to generate association rules.":
        "Lancez src/v2_assoc_rules.py pour générer les règles d'association.",
    "Run src/v2_backtest.py to generate backtest outputs.":
        "Lancez src/v2_backtest.py pour générer les sorties de backtest.",
    "Run src/v2_dl_forecast.py to generate DL forecasts.":
        "Lancez src/v2_dl_forecast.py pour générer les prévisions par apprentissage profond.",
    "Run src/v2_dtw_clusters.py to generate DTW cluster outputs.":
        "Lancez src/v2_dtw_clusters.py pour générer les sorties de clusters DTW.",
    "Run src/v2_generate_synth.py and src/v2_analytics.py for v2 outputs.":
        "Lancez src/v2_generate_synth.py et src/v2_analytics.py pour les sorties v2.",
    "Run src/v2_generate_synth.py to generate v2 datasets.":
        "Lancez src/v2_generate_synth.py pour générer les jeux de données v2.",
    "Run src/v2_graph_cluster.py to generate network outputs.":
        "Lancez src/v2_graph_cluster.py pour générer les sorties de réseau.",
    "Run src/v2_trajectory.py to generate trajectory outputs.":
        "Lancez src/v2_trajectory.py pour générer les sorties de trajectoires.",
    "Synthetic data for demonstration only.":
        "Données synthétiques à des fins de démonstration uniquement.",
    "The v3 estimator predicts high-risk category membership, not individual outcomes.":
        "L'estimateur v3 prédit l'appartenance à une catégorie de haut risque, pas des résultats individuels.",
    "This is a demo model: probabilities reflect patterns in the selected dataset, not clinical risk.":
        "Modèle de démonstration : les probabilités reflètent les schémas du jeu de données sélectionné, pas un risque clinique.",
    "This line is the population-weighted regional aggregate.":
        "Cette ligne est l'agrégat régional pondéré par la population.",
    "Tip: brush the scatter to filter the map and table.":
        "Astuce : balayez le nuage pour filtrer la carte et la table.",
    "Top drivers for this prediction in log-odds space (positive increases risk).":
        "Principaux facteurs de cette prédiction en log-odds (positif = risque accru).",
    "Uses one row per country to avoid leakage across age groups.":
        "Utilise une ligne par pays pour éviter la fuite entre groupes d'âge.",
    "scikit-learn is required for the ML demo.":
        "scikit-learn est requis pour la démo ML.",
    "In-sample (optimistic): ": "Sur échantillon (optimiste) : ",
    "Missing columns for ML demo: ": "Colonnes manquantes pour la démo ML : ",
    # --- KPI / metric labels ---
    "Countries": "Pays",
    "Avg age-std rate": "Taux standardisé moyen",
    "Median age-std rate": "Taux standardisé médian",
    "Avg suicide rate": "Taux de suicide moyen",
    "Avg risk index": "Indice de risque moyen",
    "Suicide vs Depression corr": "Corrélation suicide vs dépression",
    "Clusters (k)": "Clusters (k)",
    "Outliers flagged": "Valeurs aberrantes signalées",
    "Baseline high-risk rate": "Taux de base de haut risque",
    "Predicted high-risk probability": "Probabilité de haut risque prédite",
    "Predicted label": "Étiquette prédite",
    "High-risk": "Haut risque",
    "Not high-risk": "Pas à haut risque",
    "Accuracy": "Exactitude",
    "Accuracy (OOF)": "Exactitude (hors plis)",
    "ROC AUC": "ROC AUC",
    "ROC AUC (OOF)": "ROC AUC (hors plis)",
    "Brier (lower=better)": "Brier (plus bas = mieux)",
    "Brier OOF (lower=better)": "Brier hors plis (plus bas = mieux)",
    "Communicable, maternal, neonatal, and nutritional diseases":
        "Maladies transmissibles, maternelles, néonatales et nutritionnelles",
    "Injuries": "Traumatismes",
    "Non-communicable diseases": "Maladies non transmissibles",
    "Other causes": "Autres causes",
}


CHART_GUIDES_FR: dict[str, dict[str, str]] = {
    # --- v1 ---
    "overview_kpis": {"title": "Indicateurs clés — Aperçu", "objective": "Fournir une vérification rapide de l'échelle des taux de suicide et de la couverture.", "how": "Les indicateurs montrent le nombre de pays, la moyenne et la médiane pour les deux sexes en 2021.", "why": "Les indicateurs établissent la référence avant d'explorer les distributions."},
    "overview_top10": {"title": "Top 10 des taux de suicide", "objective": "Mettre en évidence les pays aux taux standardisés les plus élevés.", "how": "Les barres sont classées par taux ; plus la barre est longue, plus le taux est élevé.", "why": "Les vues classées facilitent la comparaison des extrêmes."},
    "overview_region_box": {"title": "Dispersion régionale", "objective": "Comparer les distributions régionales, pas seulement les moyennes.", "how": "Les boîtes montrent médianes et dispersion ; les points représentent les pays.", "why": "Les boîtes à moustaches révèlent la variabilité au sein des régions."},
    "overview_corr": {"title": "Signal de démo du modèle", "objective": "Vérifier si suicide et dépression évoluent ensemble dans la table fusionnée.", "how": "La corrélation est calculée sur 25+ ans, les deux sexes.", "why": "Cela fournit une vérification rapide avant la modélisation."},
    "who_map": {"title": "Carte OMS", "objective": "Montrer la variation géographique de la mesure sélectionnée.", "how": "Une couleur plus foncée signifie un taux plus élevé ; survolez pour les valeurs exactes.", "why": "Les cartes font ressortir rapidement les regroupements spatiaux."},
    "who_top_bottom": {"title": "Pays extrêmes (haut et bas)", "objective": "Classer les extrêmes pour la mesure sélectionnée.", "how": "Les barres de gauche montrent le top 10, celles de droite le bas 10.", "why": "Les classements côte à côte montrent la dispersion."},
    "who_sex_compare": {"title": "Comparaison par sexe", "objective": "Comparer les taux hommes vs femmes pour un pays.", "how": "Les barres montrent la mesure sélectionnée pour chaque sexe.", "why": "Les écarts entre sexes sont un schéma épidémiologique clé."},
    "who_crude_vs_age": {"title": "Brut vs standardisé par âge", "objective": "Montrer comment la structure d'âge modifie la comparaison des taux.", "how": "Les points au-dessus de la diagonale signifient un taux standardisé plus élevé.", "why": "Cela explique pourquoi taux bruts et standardisés peuvent différer."},
    "depression_map": {"title": "Carte des DALYs de dépression", "objective": "Cartographier le taux de DALYs par pays pour le groupe d'âge sélectionné.", "how": "La couleur encode le taux de DALYs ; survolez pour les valeurs exactes.", "why": "Les cartes montrent où le fardeau se concentre."},
    "depression_top20": {"title": "Top 20 des DALYs", "objective": "Classer les pays au fardeau le plus élevé pour le groupe d'âge sélectionné.", "how": "Les barres sont classées par taux de DALYs.", "why": "Les classements rendent les valeurs extrêmes visibles."},
    "depression_age_bar": {"title": "DALYs par groupe d'âge", "objective": "Comparer les taux moyens de DALYs entre groupes d'âge.", "how": "Les barres montrent la moyenne pour chaque tranche d'âge, les deux sexes.", "why": "Met en évidence le groupe d'âge portant le plus grand fardeau."},
    "addiction_map": {"title": "Carte de l'addiction", "objective": "Cartographier le taux de décès pour la cause de consommation de substances et le sexe sélectionnés.", "how": "La couleur encode le taux de décès ; survolez pour les valeurs exactes.", "why": "Les schémas spatiaux diffèrent selon la substance."},
    "addiction_top20": {"title": "Top 20 des décès par addiction", "objective": "Classer les pays au fardeau le plus élevé pour la cause et le sexe sélectionnés.", "how": "Les barres sont classées par taux de décès.", "why": "Les classements clarifient la queue de distribution."},
    "addiction_sex_compare": {"title": "Comparaison par sexe", "objective": "Comparer les taux de décès hommes vs femmes pour un pays.", "how": "Les barres montrent la moyenne pour la cause sélectionnée.", "why": "Les différences entre sexes sont souvent importantes pour les substances."},
    "selfharm_map": {"title": "Carte de l'automutilation", "objective": "Cartographier le taux de décès par automutilation pour l'âge et le sexe sélectionnés.", "how": "La couleur encode le taux de décès ; survolez pour les valeurs exactes.", "why": "Les cartes montrent rapidement la variation géographique."},
    "selfharm_top20": {"title": "Top 20 des décès par automutilation", "objective": "Classer les pays au fardeau le plus élevé pour l'âge et le sexe sélectionnés.", "how": "Les barres sont classées par taux de décès.", "why": "Les classements révèlent les extrêmes."},
    "selfharm_sex_compare": {"title": "Comparaison par sexe", "objective": "Comparer les taux d'automutilation hommes vs femmes pour un pays.", "how": "Les barres montrent le groupe d'âge sélectionné pour chaque sexe.", "why": "Les écarts entre sexes indiquent des profils de risque différents."},
    "selfharm_methods": {"title": "Répartition des méthodes", "objective": "Comparer les taux moyens par méthode, si disponible.", "how": "Les barres montrent les taux moyens par catégorie de méthode.", "why": "Les méthodes montrent comment le fardeau diffère au sein de l'automutilation."},
    "probdeath_map": {"title": "Carte de la probabilité de décès", "objective": "Cartographier la probabilité de décès pour la cause, le sexe et l'âge sélectionnés.", "how": "La couleur encode la probabilité ; les valeurs ne sont pas des taux pour 100 000.", "why": "Les probabilités montrent des niveaux de risque relatifs."},
    "probdeath_top20": {"title": "Top 20 des probabilités", "objective": "Classer les pays à la plus forte probabilité de décès.", "how": "Les barres sont classées par valeur de probabilité.", "why": "Le classement montre les pays les plus à risque."},
    "allcause_trend": {"title": "Courbe de tendance toutes causes", "objective": "Suivre l'évolution dans le temps pour la mesure sélectionnée.", "how": "La courbe montre la valeur de la mesure par année.", "why": "Les tendances montrent la direction et l'ampleur du changement."},
    "bigcat_treemap": {"title": "Treemap des grandes catégories", "objective": "Montrer comment les DALYs se répartissent entre niveaux de cause.", "how": "La taille des boîtes reflète la valeur ; la hiérarchie montre la structure parent-enfant.", "why": "Les treemaps mettent en évidence composition et hiérarchie à la fois."},
    "bigcat_donut": {"title": "Anneau des grandes catégories", "objective": "Montrer les parts des catégories de premier niveau.", "how": "Les parts représentent les catégories de niveau 1.", "why": "Les anneaux simplifient la vue de composition."},
    "relationships_scatter": {"title": "Nuage de relations", "objective": "Comparer les taux de suicide avec un indicateur sélectionné.", "how": "Chaque point est un pays ; la couleur indique la région.", "why": "Les nuages de points mettent en évidence des schémas linéaires et non linéaires."},
    "relationships_heatmap": {"title": "Carte de chaleur des corrélations", "objective": "Résumer les corrélations entre indicateurs.", "how": "Les cellules montrent les coefficients de corrélation de Pearson.", "why": "Les cartes de chaleur permettent une comparaison rapide de nombreuses paires."},
    "segmentation_map": {"title": "Carte des clusters", "objective": "Montrer la distribution géographique des clusters.", "how": "La couleur indique l'étiquette de cluster.", "why": "Le contexte spatial clarifie les schémas de clusters."},
    "segmentation_sizes": {"title": "Tailles des clusters", "objective": "Montrer combien de pays appartiennent à chaque cluster.", "how": "Les barres montrent le nombre de pays par cluster.", "why": "L'équilibre des tailles indique la stabilité des clusters."},
    "segmentation_profile": {"title": "Profils de clusters", "objective": "Comparer les schémas d'indicateurs entre clusters.", "how": "Les scores Z montrent les valeurs relatives hautes et basses.", "why": "Les profils expliquent ce qui rend un cluster distinct."},
    "segmentation_k": {"title": "Sélection de k", "objective": "Montrer les scores de silhouette pour différentes valeurs de k.", "how": "Une silhouette plus élevée suggère une meilleure séparation.", "why": "Aide à justifier le nombre de clusters choisi."},
    "outliers_scatter": {"title": "Nuage des valeurs aberrantes", "objective": "Mettre en évidence les pays aux combinaisons inhabituelles.", "how": "La taille ou la couleur signale le statut ou le score d'anomalie.", "why": "Les analyses d'anomalies soutiennent les découvertes de fouille de données."},
    "ml_results": {"title": "Métriques sur jeu de test", "objective": "Comparer l'exactitude des modèles de référence sur un jeu de test.", "how": "Une MAE plus faible et un R² plus élevé indiquent un meilleur ajustement.", "why": "Les métriques sur test montrent la performance sur données non vues."},
    "ml_cv": {"title": "Métriques de validation croisée", "objective": "Évaluer la stabilité entre plis.", "how": "La moyenne et l'écart-type montrent performance moyenne et variance.", "why": "La validation croisée réduit la sensibilité à un seul découpage."},
    "ml_pred_actual": {"title": "Prédit vs réel", "objective": "Vérifier la calibration et la dispersion des erreurs.", "how": "Les points proches de la diagonale indiquent des prédictions exactes.", "why": "Une vérification visuelle complète les métriques numériques."},
    "ml_feature_importance": {"title": "Importance des variables", "objective": "Identifier les variables qui influencent le plus le modèle.", "how": "Les barres montrent les scores d'importance RandomForest.", "why": "L'importance soutient l'interprétation et la discussion."},
    # --- v2 ---
    "v2_overview_kpis": {"title": "Indicateurs clés — Aperçu", "objective": "Fournir un contexte rapide d'échelle et de référence pour l'année et le sexe sélectionnés.", "how": "Les indicateurs résument les pays, le taux de suicide moyen et l'indice de risque moyen.", "why": "Les indicateurs posent le contexte avant une exploration plus poussée."},
    "v2_overview_map": {"title": "Carte du taux de suicide synthétique", "objective": "Montrer la variation géographique des taux de suicide pour l'année et le sexe sélectionnés.", "how": "Une couleur plus foncée indique des taux synthétiques plus élevés ; survolez pour les détails.", "why": "Les cartes exposent rapidement les schémas spatiaux."},
    "v2_overview_trend": {"title": "Tendance régionale", "objective": "Suivre l'agrégat régional dans le temps et montrer la dispersion intra-régionale.", "how": "La ligne est l'agrégat ; la bande ombrée est l'IQR entre pays.", "why": "Combine tendance centrale et dispersion."},
    "v2_overview_benchmark": {"title": "Table de benchmark des indicateurs", "objective": "Comparer les pays aux percentiles mondiaux pour la mesure choisie.", "how": "Rouge au-dessus du p90 ; bleu en dessous du p10 ; triez par valeur.", "why": "Met en évidence les extrêmes de façon standardisée."},
    "v2_clusters_map": {"title": "Carte des clusters de profils", "objective": "Montrer où apparaissent géographiquement les clusters synthétiques.", "how": "Chaque couleur est une étiquette de cluster.", "why": "Le contexte spatial aide à interpréter les profils de clusters."},
    "v2_clusters_centers": {"title": "Centres des clusters", "objective": "Résumer les indicateurs moyens par cluster.", "how": "Chaque ligne est un cluster ; les colonnes sont les valeurs moyennes des variables.", "why": "Les centres expliquent ce qui distingue les clusters."},
    "v2_clusters_k": {"title": "Sélection de k", "objective": "Documenter les scores de silhouette pour différentes valeurs de k.", "how": "Une silhouette plus élevée indique une meilleure séparation.", "why": "Justifie le nombre de clusters choisi."},
    "v2_traj_map": {"title": "Carte des clusters de trajectoires", "objective": "Montrer la distribution spatiale des clusters de trajectoires.", "how": "Chaque pays est coloré selon son cluster de trajectoire.", "why": "Met en évidence les différences géographiques des tendances à long terme."},
    "v2_traj_scatter_slope": {"title": "Pente vs volatilité", "objective": "Comparer la direction à long terme et l'instabilité.", "how": "À droite = tendances croissantes ; en haut = plus volatile.", "why": "Sépare la croissance régulière des schémas bruités."},
    "v2_traj_scatter_mean": {"title": "Taux moyen vs évolution 5 ans", "objective": "Comparer les niveaux de base avec l'élan récent.", "how": "À droite = moyenne plus élevée ; en haut = hausse récente.", "why": "Montre si les niveaux élevés augmentent ou se stabilisent."},
    "v2_traj_centers": {"title": "Centres de trajectoires", "objective": "Résumer les moyennes des variables par cluster.", "how": "Chaque ligne est un cluster ; les colonnes sont les variables de trajectoire.", "why": "Soutient le récit de chaque type de cluster."},
    "v2_traj_k": {"title": "Sélection de k", "objective": "Montrer les métriques de silhouette pour le clustering de trajectoires.", "how": "Une silhouette plus élevée indique une séparation plus forte.", "why": "Fournit la justification du clustering."},
    "v2_dtw_map": {"title": "Carte des clusters DTW", "objective": "Cartographier les clusters selon la forme des séries temporelles.", "how": "Les couleurs représentent les étiquettes de clusters DTW.", "why": "La DTW capture des formes similaires même avec des décalages temporels."},
    "v2_dtw_prototype": {"title": "Trajectoire prototype DTW", "objective": "Montrer la forme de trajectoire typique d'un cluster.", "how": "La ligne est la série temporelle centrale du cluster.", "why": "Aide à interpréter le sens de chaque cluster DTW."},
    "v2_dtw_k": {"title": "Sélection de k (DTW)", "objective": "Montrer l'inertie et la silhouette pour le clustering DTW.", "how": "Utilisez ces métriques pour équilibrer ajustement et parcimonie.", "why": "Documente le k choisi."},
    "v2_network_plot": {"title": "Réseau de similarité des pays", "objective": "Visualiser la structure de similarité entre pays.", "how": "Les nœuds sont des pays ; les liens montrent la similarité ; la taille reflète la centralité.", "why": "Les réseaux révèlent communautés et pôles."},
    "v2_network_sizes": {"title": "Tailles des communautés", "objective": "Montrer combien de pays appartiennent à chaque communauté.", "how": "Les lignes listent les clusters et leurs effectifs.", "why": "L'équilibre des tailles indique la structure communautaire."},
    "v2_network_central": {"title": "Pays les plus centraux", "objective": "Identifier les pôles à forte intermédiarité ou degré.", "how": "Une centralité plus élevée signifie plus de connexions ou de pontage.", "why": "Met en évidence les nœuds influents du réseau."},
    "v2_network_edges": {"title": "Liens les plus forts", "objective": "Lister les paires de pays les plus similaires.", "how": "Une similarité plus élevée signifie des profils plus proches.", "why": "Montre les liens les plus forts du réseau."},
    "v2_linked_scatter": {"title": "Nuage lié", "objective": "Permettre la sélection par balayage pour filtrer le reste de la vue.", "how": "Sélectionnez des points ; la carte et la table se mettent à jour.", "why": "Les vues liées soutiennent l'exploration interactive."},
    "v2_linked_map": {"title": "Carte filtrée", "objective": "Montrer les pays sélectionnés dans leur contexte géographique.", "how": "La couleur encode la mesure y choisie.", "why": "Les cartes aident à valider les schémas spatiaux de la sélection."},
    "v2_linked_table": {"title": "Table filtrée", "objective": "Inspecter les pays sélectionnés et leurs valeurs.", "how": "La table est triée par la mesure de l'axe y.", "why": "La vue tabulaire soutient une inspection détaillée."},
    "v2_linked_multiples": {"title": "Petits multiples régionaux", "objective": "Comparer les tendances régionales côte à côte.", "how": "Chaque panneau est une région ; l'axe y est la mesure sélectionnée.", "why": "Les petits multiples facilitent les comparaisons."},
    "v2_forecast_line": {"title": "Courbe de prévision", "objective": "Comparer réel vs prévu par région.", "how": "Des couleurs différentes indiquent les segments réel et prévu.", "why": "Montre les projections du modèle aux côtés de l'historique."},
    "v2_forecast_metrics": {"title": "Métriques de prévision (DL)", "objective": "Résumer l'erreur du modèle pour les prévisions par apprentissage profond.", "how": "Une MAE/RMSE plus faible indique un meilleur ajustement.", "why": "Quantifie la performance de prévision."},
    "v2_backtest_line": {"title": "Courbe de backtest", "objective": "Comparer prédit vs réel dans un dispositif glissant.", "how": "Un alignement plus proche indique un meilleur ajustement du backtest.", "why": "Teste la stabilité entre fenêtres temporelles."},
    "v2_backtest_metrics": {"title": "Métriques de backtest", "objective": "Résumer l'erreur du backtest par région.", "how": "Une erreur plus faible signifie une meilleure généralisation.", "why": "Soutient la validation du modèle."},
    "v2_scenario_pred": {"title": "Prédiction de scénario", "objective": "Montrer le taux de suicide prédit selon les entrées actuelles.", "how": "La prédiction se met à jour instantanément quand les curseurs changent.", "why": "Soutient l'analyse hypothétique."},
    "v2_sensitivity": {"title": "Graphique de sensibilité", "objective": "Montrer l'effet d'une variation de 10 % d'une variable.", "how": "Les barres représentent le pourcentage de changement de la prédiction.", "why": "Met en évidence l'influence locale de chaque variable."},
    "v2_quantile": {"title": "Bandes de prédiction par quantiles", "objective": "Montrer les bandes d'incertitude autour des prédictions.", "how": "La bande est q10-q90 ; q50 est la prédiction médiane.", "why": "Communique l'incertitude, pas seulement des estimations ponctuelles."},
    "v2_quantile_metrics": {"title": "Métriques quantiles", "objective": "Résumer l'exactitude du modèle quantile.", "how": "Utilisez-les comme indicateurs de diagnostic.", "why": "Valide le modèle d'intervalles."},
    "v2_explain_perm": {"title": "Importance par permutation", "objective": "Expliquer quelles variables comptent le plus pour le modèle.", "how": "Des valeurs plus grandes signifient une plus forte baisse de performance après mélange.", "why": "Soutient l'interprétabilité du modèle."},
    "v2_explain_pdp": {"title": "Dépendance partielle", "objective": "Montrer comment les prédictions changent selon les valeurs d'une variable.", "how": "Les lignes montrent l'effet marginal par variable.", "why": "Clarifie les schémas de réponse non linéaires."},
    "v2_outliers_scatter": {"title": "Nuage des valeurs aberrantes", "objective": "Mettre en évidence les pays anormaux dans l'espace des variables.", "how": "Les valeurs aberrantes sont signalées et dimensionnées par score.", "why": "Soutient les découvertes de détection d'anomalies."},
    "v2_outliers_table": {"title": "Principales anomalies", "objective": "Lister les valeurs aberrantes au score le plus élevé.", "how": "Les lignes montrent les pays et leurs raisons d'anomalie.", "why": "Fournit une liste concrète à examiner."},
    "v2_patterns_table": {"title": "Règles d'association", "objective": "Montrer les schémas de cooccurrence les plus forts.", "how": "Triez par lift et examinez antécédents vers conséquents.", "why": "Révèle les structures de cooccurrence."},
    "v2_patterns_interp": {"title": "Interprétation des règles", "objective": "Expliquer la règle principale en mots.", "how": "Lisez antécédents -> conséquents avec lift et confiance.", "why": "Améliore l'interprétabilité pour les lecteurs non techniques."},
}


# Templates with {} placeholders — see i18n.fmt().
FMT_FR: dict[str, str] = {
    "Missing {}: {}": "Manquant {} : {}",
    "Missing {}. Run {}.": "Fichier manquant : {}. Lancez {}.",
    "Manifest missing columns: {}": "Colonnes manquantes du manifeste : {}",
    "**Dataset:** {}": "**Jeu de données :** {}",
    "**Chart type:** {}": "**Type de graphique :** {}",
    "**Objective:** {}": "**Objectif :** {}",
    "**Result:** {}": "**Résultat :** {}",
    "**Why this chart:** {}": "**Pourquoi ce graphique :** {}",
    "**Key takeaway:** {}": "**À retenir :** {}",
    "Red = above p90 ({}), Blue = below p10 ({}).":
        "Rouge = au-dessus du p90 ({}), Bleu = en dessous du p10 ({}).",
    "Selection: {} countries.": "Sélection : {} pays.",
    "### Predicted suicide rate: **{}**": "### Taux de suicide prédit : **{}**",
    "When {} then {} ": "Quand {} alors {} ",
    "Region: {} | Income group: {}": "Région : {} | Groupe de revenu : {}",
    "High-risk means suicide_rate ≥ {} per 100k ":
        "Haut risque signifie taux_suicide ≥ {} pour 100 000 ",
    "### Model diagnostics (out-of-fold, {}-fold CV)":
        "### Diagnostics du modèle (hors plis, CV {} plis)",
    "Brier score (train): {}": "Score de Brier (entraînement) : {}",
    "(lift {}, confidence {}).": "(lift {}, confiance {}).",
    "(p{} of the selected dataset).": "(p{} du jeu de données sélectionné).",
}
