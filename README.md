# Projet de Machine Learning pour la Prédiction du Risque de Diabète

Ce projet est une application de machine learning complète conçue pour prédire le risque de diabète de type 2. Il couvre l'ensemble du cycle de vie du machine learning, de l'exploration des données et de l'entraînement du modèle au suivi des expériences, au déploiement et à la création d'une interface web conviviale pour l'aide à la décision clinique.

Le projet utilise la base de données sur le diabète des Indiens Pima. Le flux de travail principal implique une analyse exploratoire des données, l'entraînement et l'évaluation de plusieurs modèles de classification, un suivi rigoureux des expériences avec MLflow, et enfin la mise à disposition d'un tableau de bord interactif.

## Fonctionnalités Clés

-   **Analyse Exploratoire des Données (EDA) :** Le notebook `EDA.ipynb` effectue une exploration approfondie du jeu de données `diabetes.csv`, y compris le nettoyage des valeurs manquantes (zéros impossibles remplacés par des NaNs) et l'enregistrement du jeu de données nettoyé (`save_final.csv`).
-   **Entraînement et Évaluation de Modèles avec MLflow :** Le notebook `diabetes_mlflow.ipynb` est le cœur du projet. Il gère :
    -   Le chargement du jeu de données nettoyé (`save_final.csv`).
    -   La préparation des données (imputation, mise à l'échelle) via des pipelines `scikit-learn`.
    -   L'entraînement et l'optimisation de modèles (Régression Logistique, Random Forest, XGBoost) à l'aide de `GridSearchCV`.
    -   L'intégration complète de MLflow pour le suivi des expériences, l'enregistrement des paramètres, des métriques (ROC AUC, Recall, F1-Score) et la versionisation des modèles.
    -   La sélection du meilleur modèle et sa sauvegarde sous forme de fichier `.joblib` (`diabetes_risk_model.joblib`).
    -   Un exemple de code pour le déploiement via FastAPI.
-   **Tableau de Bord Interactif Streamlit :** L'application `dashborad_kpi.py` fournit une interface utilisateur intuitive pour les professionnels de la santé. Elle charge le modèle sauvegardé et permet :
    -   La saisie des caractéristiques du patient via des sliders.
    -   La prédiction instantanée du risque de diabète.
    -   L'affichage de métriques de performance du modèle et de graphiques explicatifs (importance des caractéristiques, analyse de la courbe de décision, courbes de calibration).
    -   Un support à la décision clinique basé sur le risque prédit.
-   **Exemple d'Expérimentation Modèle (sans MLflow) :** Le notebook `main.ipynb` est une version alternative ou plus ancienne du processus d'expérimentation des modèles, incluant l'entraînement, l'évaluation et l'optimisation de seuils cliniques, mais sans l'intégration de MLflow.

## Technologies Utilisées

-   **Analyse et Manipulation de Données :** `pandas`, `numpy`
-   **Machine Learning :** `scikit-learn`, `xgboost`
-   **MLOps :** `mlflow`
-   **Visualisation :** `matplotlib`, `seaborn`, `shap`
-   **Web Dashboard :** `streamlit`
-   **API (exemple) :** `fastapi`, `uvicorn`, `pydantic`
-   **Utilitaires :** `ipykernel`, `joblib`, `kagglehub` (pour l'accès potentiel aux données Kaggle)

## Démarrage Rapide

### Prérequis

-   Python 3.13
-   Pip

### Installation

1.  Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-nom-d-utilisateur/votre-dépôt.git
    ```
2.  Créez et activez un environnement virtuel (recommandé) :
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  Installez toutes les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```

### Utilisation

1.  **Exécutez l'Analyse Exploratoire des Données (EDA) :**
    Télecharger la dataset : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv
    et mettez le dans le dossier 
    Ouvrez et exécutez le notebook `EDA.ipynb` dans Jupyter Lab ou Jupyter Notebook pour nettoyer le jeu de données et générer `save_final.csv`.
    ```bash
    pip install jupyterlab
    jupyter lab
    ```

2.  **Exécutez l'Entraînement du Modèle avec MLflow :**
    Ouvrez et exécutez le notebook `diabetes_mlflow.ipynb` dans Jupyter Lab pour entraîner les modèles, optimiser les hyperparamètres et suivre les expériences avec MLflow. Cela sauvegardera également le meilleur modèle sous `diabetes_risk_model.joblib`.

3.  **Exécutez le Tableau de Bord Streamlit :**
    Pour démarrer le tableau de bord interactif, exécutez la commande suivante dans votre terminal :
    ```bash
    streamlit run dashborad_kpi.py
    ```

4.  **Visualisez les Expériences MLflow :**
    Pour accéder à l'interface de suivi des expériences MLflow, exécutez :
    ```bash
    mlflow ui
    ```

## Structure du Projet

```
.
├── dashborad_kpi.py            # Application de tableau de bord Streamlit
├── diabetes_mlflow.ipynb       # Jupyter Notebook principal pour l'entraînement de modèles et les expériences MLflow
├── diabetes_risk_model.joblib  # Modèle de machine learning sauvegardé (meilleur modèle)
├── diabetes.csv                # Jeu de données brut sur le diabète
├── EDA.ipynb                   # Jupyter Notebook pour l'Analyse Exploratoire des Données
├── main.ipynb                  # Jupyter Notebook alternatif/précédent pour l'expérimentation de modèles (sans MLflow)
├── mlflow.db                   # Base de données MLflow pour le suivi des expériences
├── requirements.txt            # Liste des dépendances Python du projet
├── roc_curve.png               # Courbe ROC générée (artefact)
├── save_final.csv              # Jeu de données nettoyé résultant de l'EDA
├── shap_summary.png            # Résumé SHAP généré (artefact)
└── mlruns/                     # Répertoire contenant les données de suivi des expériences MLflow
```
