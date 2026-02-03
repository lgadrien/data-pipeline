"""
================================================================================
                        ETL & TRAINING PIPELINE
================================================================================

DESCRIPTION:
    Ce script est le cœur du pipeline de données. Il s'exécute une seule fois
    au démarrage du projet (dans un container Docker) et effectue 3 tâches :

    1. EXTRACT : Lit les données brutes depuis le fichier CSV
    2. LOAD    : Envoie les données dans PostgreSQL
    3. TRAIN   : Entraîne un modèle ML et le sauvegarde

FLUX D'EXÉCUTION:
    ┌─────────────────┐
    │   iris.csv      │  ← Fichier source (150 fleurs Iris)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    EXTRACT      │  ← Lecture avec Pandas
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │     LOAD        │  ← Insertion dans PostgreSQL
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │     TRAIN       │  ← Entraînement RandomForest
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  iris_model.joblib  │  ← Modèle sauvegardé (utilisé par l'API)
    └─────────────────┘

DÉPENDANCES:
    - pandas       : Manipulation des données
    - sqlalchemy   : Connexion à PostgreSQL
    - scikit-learn : Algorithme de Machine Learning
    - mlflow       : Tracking des expériences
    - joblib       : Sérialisation du modèle

AUTEUR: Projet Data Pipeline - Epitech 2025-2026
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os          # Accès aux variables d'environnement et système de fichiers
import time        # Pour attendre entre les tentatives de connexion
import pandas as pd   # Manipulation de données (DataFrames)
import numpy as np    # Calculs numériques
from sqlalchemy import create_engine, text  # Connexion et requêtes PostgreSQL
import mlflow         # Plateforme de tracking ML
import mlflow.sklearn # Extension MLflow pour scikit-learn
from sklearn.model_selection import train_test_split  # Séparation train/test
from sklearn.ensemble import RandomForestRegressor    # Algorithme de ML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Métriques
import joblib  # Sauvegarde/chargement du modèle


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Les valeurs sont récupérées depuis les variables d'environnement Docker.
# Si non définies, des valeurs par défaut sont utilisées pour le dev local.

# --- Configuration PostgreSQL ---
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")        # Utilisateur BDD
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin") # Mot de passe BDD
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")           # Nom du service Docker
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")         # Port PostgreSQL
POSTGRES_DB = os.getenv("POSTGRES_DB", "datapipeline")     # Nom de la base

# --- Configuration MLflow ---
# URL du serveur MLflow pour le tracking des expériences
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --- URL de connexion PostgreSQL ---
# Format: postgresql://user:password@host:port/database
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


# ==============================================================================
# FONCTIONS D'ATTENTE DES SERVICES
# ==============================================================================
# Ces fonctions sont essentielles car Docker ne garantit pas l'ordre de démarrage.
# Le pipeline doit attendre que PostgreSQL et MLflow soient prêts avant de continuer.

def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """
    Attend que PostgreSQL soit prêt à recevoir des connexions.

    POURQUOI C'EST NÉCESSAIRE:
        Docker démarre les containers en parallèle. Même si PostgreSQL démarre
        avant le pipeline, il lui faut quelques secondes pour être opérationnel.
        Sans cette attente, le pipeline échouerait immédiatement.

    FONCTIONNEMENT:
        1. Tente de se connecter à PostgreSQL
        2. Exécute une requête simple "SELECT 1" pour vérifier
        3. Si échec, attend 'delay' secondes et réessaie
        4. Abandonne après 'max_retries' tentatives

    Args:
        max_retries: Nombre maximum de tentatives (défaut: 30)
        delay: Délai entre chaque tentative en secondes (défaut: 2)

    Returns:
        True si la connexion réussit, False sinon
    """
    print(f"⏳ Attente de PostgreSQL ({POSTGRES_HOST}:{POSTGRES_PORT})...")

    for i in range(max_retries):
        try:
            # Création d'une connexion temporaire
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                # Requête simple pour tester la connexion
                conn.execute(text("SELECT 1"))
            print("   ✅ PostgreSQL est prêt!")
            engine.dispose()  # Libère la connexion
            return True
        except Exception as e:
            # La connexion a échoué, on attend et on réessaie
            print(f"   ⏳ Tentative {i+1}/{max_retries}...")
            time.sleep(delay)

    print("   ❌ PostgreSQL n'est pas disponible")
    return False


def wait_for_mlflow(max_retries: int = 30, delay: int = 2) -> bool:
    """
    Attend que le serveur MLflow soit prêt.

    POURQUOI C'EST NÉCESSAIRE:
        MLflow doit être opérationnel pour enregistrer les métriques et le modèle.
        On vérifie sa disponibilité via son endpoint /health.

    FONCTIONNEMENT:
        1. Envoie une requête HTTP GET à l'endpoint /health de MLflow
        2. Si réponse OK (200), MLflow est prêt
        3. Sinon, attend et réessaie

    Args:
        max_retries: Nombre maximum de tentatives
        delay: Délai entre chaque tentative en secondes

    Returns:
        True si MLflow répond, False sinon
    """
    import urllib.request
    import urllib.error

    print(f"⏳ Attente de MLflow ({MLFLOW_TRACKING_URI})...")

    for i in range(max_retries):
        try:
            # Requête HTTP simple vers l'endpoint de santé
            urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
            print("   ✅ MLflow est prêt!")
            return True
        except (urllib.error.URLError, Exception):
            print(f"   ⏳ Tentative {i+1}/{max_retries}...")
            time.sleep(delay)

    print("   ❌ MLflow n'est pas disponible")
    return False


# ==============================================================================
# ÉTAPE 1: EXTRACT (EXTRACTION DES DONNÉES)
# ==============================================================================

def extract_data(filepath: str) -> pd.DataFrame:
    """
    Lit les données depuis un fichier CSV et les charge dans un DataFrame.

    C'EST QUOI UN DATAFRAME:
        Un DataFrame est une structure de données tabulaire (comme un tableau Excel)
        fournie par la librairie Pandas. Chaque colonne a un nom et un type.

    CONTENU DU FICHIER iris.csv:
        - sepal_length : Longueur des sépales (cm) ← CE QU'ON VEUT PRÉDIRE
        - sepal_width  : Largeur des sépales (cm)  ← NOTRE VARIABLE D'ENTRÉE
        - petal_length : Longueur des pétales (cm)
        - petal_width  : Largeur des pétales (cm)
        - species      : Espèce de la fleur (setosa, versicolor, virginica)

    Args:
        filepath: Chemin vers le fichier CSV

    Returns:
        DataFrame Pandas contenant les données
    """
    print(f"📥 Lecture du fichier: {filepath}")

    # Lecture du CSV - Pandas détecte automatiquement les colonnes et types
    df = pd.read_csv(filepath)

    # Affichage des informations sur les données chargées
    print(f"   ✅ {len(df)} lignes chargées")
    print(f"   📊 Colonnes: {list(df.columns)}")

    return df


# ==============================================================================
# ÉTAPE 2: LOAD (CHARGEMENT DANS POSTGRESQL)
# ==============================================================================

def load_to_postgres(df: pd.DataFrame, table_name: str = "iris_data") -> None:
    """
    Envoie le DataFrame dans une table PostgreSQL.

    POURQUOI STOCKER DANS UNE BASE DE DONNÉES:
        1. Persistance : Les données survivent aux redémarrages
        2. Requêtage : On peut interroger les données avec SQL
        3. Intégration : D'autres services peuvent accéder aux données
        4. Historique : On garde une trace des données d'entraînement

    FONCTIONNEMENT:
        1. Crée une connexion à PostgreSQL via SQLAlchemy
        2. Utilise df.to_sql() pour créer la table et insérer les données
        3. if_exists="replace" : Supprime et recrée la table à chaque exécution

    Args:
        df: DataFrame contenant les données à insérer
        table_name: Nom de la table de destination (défaut: "iris_data")
    """
    print(f"📤 Envoi des données vers PostgreSQL (table: {table_name})")

    # Création de la connexion
    engine = create_engine(DATABASE_URL)

    # Insertion des données
    # if_exists="replace" : Si la table existe, on la supprime et on la recrée
    # index=False : On n'insère pas l'index du DataFrame comme colonne
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    # Vérification : On compte les lignes insérées
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()

    print(f"   ✅ {count} lignes insérées dans la table '{table_name}'")

    # Libération de la connexion
    engine.dispose()


# ==============================================================================
# ÉTAPE 3: TRAIN (ENTRAÎNEMENT DU MODÈLE)
# ==============================================================================

def train_model(df: pd.DataFrame) -> dict:
    """
    Entraîne un modèle RandomForestRegressor sur les données Iris.

    OBJECTIF:
        Prédire sepal_length (longueur des sépales) à partir de sepal_width (largeur)
        C'est un problème de RÉGRESSION (prédire une valeur continue, pas une catégorie)

    ALGORITHME UTILISÉ - RandomForest:
        - Ensemble de 100 arbres de décision
        - Chaque arbre apprend sur un échantillon différent des données
        - La prédiction finale = moyenne des prédictions de tous les arbres
        - Avantages : Robuste, peu de réglages nécessaires, bon pour les petits datasets

    MÉTRIQUES CALCULÉES:
        - RMSE (Root Mean Square Error) : Erreur quadratique moyenne, pénalise les grandes erreurs
        - MAE (Mean Absolute Error) : Erreur absolue moyenne, plus intuitive
        - R² Score : Coefficient de détermination, mesure la qualité de l'ajustement (1 = parfait)

    MLFLOW TRACKING:
        Tout est enregistré dans MLflow pour le suivi :
        - Paramètres du modèle (n_estimators, max_depth...)
        - Métriques de performance (RMSE, MAE, R²)
        - Le modèle lui-même (pour le récupérer plus tard)

    Args:
        df: DataFrame avec les données d'entraînement

    Returns:
        Dictionnaire contenant les métriques calculées
    """
    print("🤖 Entraînement du modèle (Régression)...")

    # --- Configuration de MLflow ---
    # On dit à MLflow où envoyer les données (serveur MLflow dans Docker)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Nom de l'expérience (groupe de "runs" liés)
    mlflow.set_experiment("iris-regression")

    # --- Préparation des données ---
    # X = Features (variables d'entrée) - ici juste sepal_width
    # y = Target (variable à prédire) - ici sepal_length
    # .values convertit en array NumPy (format attendu par scikit-learn)
    X = df[["sepal_width"]].values  # Double crochets = DataFrame 2D → Array 2D
    y = df["sepal_length"].values   # Simple crochets = Series 1D → Array 1D

    # --- Séparation Train/Test ---
    # 80% des données pour l'entraînement, 20% pour le test
    # random_state=42 : Graine aléatoire fixe pour reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Définition des hyperparamètres ---
    # Ce sont les "réglages" de l'algorithme
    params = {
        "n_estimators": 100,      # Nombre d'arbres dans la forêt
        "max_depth": 5,           # Profondeur max de chaque arbre (évite le surapprentissage)
        "min_samples_split": 2,   # Min échantillons pour diviser un nœud
        "min_samples_leaf": 1,    # Min échantillons dans une feuille
        "random_state": 42        # Reproductibilité
    }

    # --- Entraînement avec tracking MLflow ---
    # Un "run" = une exécution d'entraînement avec ses paramètres et résultats
    with mlflow.start_run(run_name="random_forest_regression"):

        # 1) Logger les paramètres (pour pouvoir les retrouver plus tard)
        mlflow.log_params(params)
        mlflow.log_param("feature", "sepal_width")
        mlflow.log_param("target", "sepal_length")

        # 2) Créer et entraîner le modèle
        # **params = décompresse le dictionnaire en arguments nommés
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)  # L'entraînement proprement dit

        # 3) Évaluer le modèle sur les données de test
        y_pred = model.predict(X_test)

        # 4) Calculer les métriques
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),  # Erreur quadratique
            "mae": mean_absolute_error(y_test, y_pred),           # Erreur absolue
            "r2_score": r2_score(y_test, y_pred)                  # Coefficient R²
        }

        # 5) Logger les métriques dans MLflow
        mlflow.log_metrics(metrics)

        # 6) Sauvegarder le modèle localement (pour l'API)
        # Ce fichier sera accessible par l'API via un volume Docker partagé
        model_dir = "/app/models"
        os.makedirs(model_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas
        model_path = os.path.join(model_dir, "iris_model.joblib")
        joblib.dump(model, model_path)  # Sérialisation du modèle

        # 7) Logger le chemin et le modèle dans MLflow
        mlflow.log_param("model_path", model_path)
        mlflow.sklearn.log_model(model, "model", registered_model_name="IrisModel")

        # 8) Récupérer l'ID du run pour référence
        run_id = mlflow.active_run().info.run_id

        # --- Affichage des résultats ---
        print(f"   ✅ Modèle entraîné avec succès!")
        print(f"   📊 Métriques:")
        for name, value in metrics.items():
            print(f"      - {name}: {value:.4f}")
        print(f"   💾 Modèle sauvegardé: {model_path}")
        print(f"   🔗 MLflow Run ID: {run_id}")

    return metrics


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    """
    Point d'entrée du pipeline - Orchestre toutes les étapes.

    SÉQUENCE D'EXÉCUTION:
        1. Attendre que PostgreSQL soit prêt
        2. Attendre que MLflow soit prêt
        3. Vérifier que le fichier CSV existe
        4. Extraire les données (EXTRACT)
        5. Charger dans PostgreSQL (LOAD)
        6. Entraîner le modèle (TRAIN)

    GESTION DES ERREURS:
        Si une étape échoue, le script s'arrête avec exit(1)
        Docker détectera l'échec et pourra relancer le container
    """
    print("=" * 50)
    print("🚀 Démarrage du pipeline ETL & Training")
    print("=" * 50)

    # --- Étape préliminaire : Attendre les dépendances ---
    if not wait_for_db():
        print("❌ Impossible de se connecter à PostgreSQL. Abandon.")
        exit(1)  # Code de sortie 1 = erreur

    if not wait_for_mlflow():
        print("❌ Impossible de se connecter à MLflow. Abandon.")
        exit(1)

    # --- Vérification du fichier source ---
    csv_path = "/app/data/iris.csv"  # Chemin dans le container Docker

    if not os.path.exists(csv_path):
        print(f"❌ Fichier non trouvé: {csv_path}")
        exit(1)

    # --- Exécution du pipeline ETL ---

    # 1. EXTRACT : Lecture du CSV
    df = extract_data(csv_path)

    # 2. LOAD : Insertion dans PostgreSQL
    load_to_postgres(df)

    # 3. TRAIN : Entraînement et sauvegarde du modèle
    metrics = train_model(df)

    print("=" * 50)
    print("✅ Pipeline terminé avec succès!")
    print("=" * 50)


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================
# Cette condition vérifie si le script est exécuté directement (pas importé)

if __name__ == "__main__":
    main()
