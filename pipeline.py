"""
ETL & Training Pipeline
========================
Service Docker indépendant qui:
1. Attend que PostgreSQL soit prêt
2. Charge iris.csv
3. Envoie les données dans PostgreSQL
4. Entraîne un RandomForestRegressor (prédit sepal_length à partir de sepal_width)
5. Logue le modèle et les métriques dans MLflow
"""

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuration via variables d'environnement
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")  # Nom du service Docker
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "datapipeline")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Connexion PostgreSQL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """Attend que la base de données soit prête"""
    print(f"⏳ Attente de PostgreSQL ({POSTGRES_HOST}:{POSTGRES_PORT})...")

    for i in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("   ✅ PostgreSQL est prêt!")
            engine.dispose()
            return True
        except Exception as e:
            print(f"   ⏳ Tentative {i+1}/{max_retries}...")
            time.sleep(delay)

    print("   ❌ PostgreSQL n'est pas disponible")
    return False


def wait_for_mlflow(max_retries: int = 30, delay: int = 2) -> bool:
    """Attend que MLflow soit prêt"""
    import urllib.request
    import urllib.error

    print(f"⏳ Attente de MLflow ({MLFLOW_TRACKING_URI})...")

    for i in range(max_retries):
        try:
            urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
            print("   ✅ MLflow est prêt!")
            return True
        except (urllib.error.URLError, Exception):
            print(f"   ⏳ Tentative {i+1}/{max_retries}...")
            time.sleep(delay)

    print("   ❌ MLflow n'est pas disponible")
    return False


def extract_data(filepath: str) -> pd.DataFrame:
    """Étape 1: Extraction - Lit le fichier CSV"""
    print(f"📥 Lecture du fichier: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   ✅ {len(df)} lignes chargées")
    print(f"   📊 Colonnes: {list(df.columns)}")
    return df


def load_to_postgres(df: pd.DataFrame, table_name: str = "iris_data") -> None:
    """Étape 2: Load - Envoie les données dans PostgreSQL"""
    print(f"📤 Envoi des données vers PostgreSQL (table: {table_name})")

    engine = create_engine(DATABASE_URL)

    # Créer/remplacer la table
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    # Vérification
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()

    print(f"   ✅ {count} lignes insérées dans la table '{table_name}'")
    engine.dispose()


def train_model(df: pd.DataFrame) -> dict:
    """
    Étape 3: Training - Entraîne un RandomForestRegressor
    Prédit sepal_length à partir de sepal_width
    """
    print("🤖 Entraînement du modèle (Régression)...")

    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris-regression")

    # Préparation des données pour la régression
    # X = sepal_width, y = sepal_length
    X = df[["sepal_width"]].values
    y = df["sepal_length"].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparamètres
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    # Entraînement avec MLflow
    with mlflow.start_run(run_name="random_forest_regression"):
        # Log des paramètres
        mlflow.log_params(params)
        mlflow.log_param("feature", "sepal_width")
        mlflow.log_param("target", "sepal_length")

        # Entraînement
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Métriques de régression
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }

        # Log des métriques
        mlflow.log_metrics(metrics)

        # Sauvegarde du modèle localement (volume partagé avec l'API)
        model_dir = "/app/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "iris_model.joblib")
        joblib.dump(model, model_path)

        # Log du chemin du modèle
        mlflow.log_param("model_path", model_path)

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="IrisModel")

        run_id = mlflow.active_run().info.run_id

        print(f"   ✅ Modèle entraîné avec succès!")
        print(f"   📊 Métriques:")
        for name, value in metrics.items():
            print(f"      - {name}: {value:.4f}")
        print(f"   💾 Modèle sauvegardé: {model_path}")
        print(f"   🔗 MLflow Run ID: {run_id}")

    return metrics


def main():
    """Pipeline principal ETL + Training"""
    print("=" * 50)
    print("🚀 Démarrage du pipeline ETL & Training")
    print("=" * 50)

    # Attendre les services
    if not wait_for_db():
        print("❌ Impossible de se connecter à PostgreSQL. Abandon.")
        exit(1)

    if not wait_for_mlflow():
        print("❌ Impossible de se connecter à MLflow. Abandon.")
        exit(1)

    # Chemin vers le fichier iris.csv (dans le volume monté)
    csv_path = "/app/data/iris.csv"

    if not os.path.exists(csv_path):
        print(f"❌ Fichier non trouvé: {csv_path}")
        exit(1)

    # 1. Extract
    df = extract_data(csv_path)

    # 2. Load to PostgreSQL
    load_to_postgres(df)

    # 3. Train & Log to MLflow
    metrics = train_model(df)

    print("=" * 50)
    print("✅ Pipeline terminé avec succès!")
    print("=" * 50)


if __name__ == "__main__":
    main()
