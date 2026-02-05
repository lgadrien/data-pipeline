"""
Model Training Script for Iris Dataset
=======================================
Uses polynomial features to improve predictions from sepal_width alone.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import joblib


def get_db_config():
    return {
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'host': os.getenv('POSTGRES_HOST', 'db'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'datapipeline')
    }


def wait_for_database(engine, max_retries=10, retry_interval=5):
    print("🔄 Waiting for database...")
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ Database connected")
            return True
        except OperationalError as e:
            print(f"⏳ Attempt {attempt}/{max_retries}...")
            if attempt == max_retries:
                raise e
            time.sleep(retry_interval)
    return False


def main():
    print("=" * 50)
    print("🚀 IRIS MODEL TRAINING (Polynomial Features)")
    print("=" * 50)

    # Database
    config = get_db_config()
    engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    wait_for_database(engine)

    # Load data
    print("\n📂 Loading data...")
    df = pd.read_sql("SELECT * FROM iris_data", engine)
    print(f"   → {len(df)} rows loaded")

    # Prepare features (sepal_width -> sepal_length)
    X = df[['sepal_width']]
    y = df['sepal_length']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   → Train: {len(X_train)}, Test: {len(X_test)}")

    # Create pipeline with polynomial features
    # This creates: sepal_width, sepal_width², sepal_width³
    print("\n🤖 Training with Polynomial Features (degree=3)...")
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'r2_score': round(r2_score(y_test, y_pred), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        'mae': round(float(mean_absolute_error(y_test, y_pred)), 4),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'model_type': 'Pipeline(PolynomialFeatures + RandomForest)',
        'polynomial_degree': 3,
        'n_estimators': 200,
        'input_feature': 'sepal_width',
        'target': 'sepal_length'
    }

    print(f"\n📈 Results:")
    print(f"   → R²: {metrics['r2_score']}")
    print(f"   → RMSE: {metrics['rmse']}")
    print(f"   → MAE: {metrics['mae']}")

    # Save model and metrics
    os.makedirs('/models', exist_ok=True)
    joblib.dump(model, '/models/model.pkl')

    with open('/models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Model saved to /models/model.pkl")
    print(f"💾 Metrics saved to /models/metrics.json")

    print("\n✅ TRAINING COMPLETE")


if __name__ == "__main__":
    main()
