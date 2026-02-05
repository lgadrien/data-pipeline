"""
Model Training Script for Iris Dataset
=======================================
This script performs the following operations:
1. Load environment variables for database access
2. Wait for PostgreSQL to be ready
3. Fetch preprocessed data from the database
4. Train a RandomForest model to predict sepal_length
5. Evaluate model performance (R², RMSE, MAE)
6. Save the trained model using joblib
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import joblib


# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT
# =============================================================================

def get_db_config():
    """Load database configuration from environment variables."""
    return {
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'host': os.getenv('POSTGRES_HOST', 'db'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'datapipeline')
    }


def build_connection_url(config):
    """Build PostgreSQL connection URL from config."""
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"


def wait_for_database(engine, max_retries=10, retry_interval=5):
    """
    Wait Strategy: Attempt to connect to the database with retries.
    """
    print("🔄 Waiting for database to be ready...")

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ Database connected successfully on attempt {attempt}")
            return True
        except OperationalError as e:
            print(f"⏳ Attempt {attempt}/{max_retries}: Database not ready. Retrying in {retry_interval}s...")
            if attempt == max_retries:
                print(f"❌ Failed to connect after {max_retries} attempts")
                raise e
            time.sleep(retry_interval)

    return False


# =============================================================================
# 2. DATA RETRIEVAL
# =============================================================================

def fetch_training_data(engine, table_name='iris_data'):
    """Fetch preprocessed data from PostgreSQL."""
    print(f"📂 Fetching data from table '{table_name}'...")

    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)

    print(f"   → Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


# =============================================================================
# 3. FEATURE PREPARATION
# =============================================================================

def prepare_features(df, target_column='sepal_length'):
    """Separate features (X) and target (y)."""
    print("🔧 Preparing features and target...")

    # Features: all numeric columns except target
    feature_columns = ['sepal_width', 'petal_length', 'petal_width']

    X = df[feature_columns]
    y = df[target_column]

    print(f"   → Features (X): {feature_columns}")
    print(f"   → Target (y): {target_column}")
    print(f"   → X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_columns


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print(f"📊 Splitting data (test_size={test_size})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"   → Training set: {len(X_train)} samples")
    print(f"   → Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 4. MODEL TRAINING
# =============================================================================

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a RandomForest Regressor model."""
    print(f"🤖 Training RandomForestRegressor (n_estimators={n_estimators})...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)

    print("   → Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics."""
    print("📈 Evaluating model performance...")

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"   → R² Score: {r2:.4f}")
    print(f"   → RMSE: {rmse:.4f}")
    print(f"   → MAE: {mae:.4f}")

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def get_feature_importance(model, feature_names):
    """Display feature importance."""
    print("🎯 Feature Importance:")

    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"   → {name}: {importance:.4f}")

    return dict(zip(feature_names, importances))


# =============================================================================
# 5. MODEL SAVING
# =============================================================================

def save_model(model, filepath='/models/model.pkl'):
    """Save the trained model using joblib."""
    print(f"💾 Saving model to '{filepath}'...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    joblib.dump(model, filepath)

    # Verify save
    file_size = os.path.getsize(filepath)
    print(f"✅ Model saved successfully ({file_size / 1024:.2f} KB)")

    return filepath


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline execution."""
    print("=" * 60)
    print("🚀 IRIS MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Configuration
    config = get_db_config()
    print(f"\n📌 Database: {config['host']}:{config['port']}/{config['database']}")

    connection_url = build_connection_url(config)
    engine = create_engine(connection_url)

    # 2. Wait for database
    wait_for_database(engine)

    # 3. Fetch data
    print("\n" + "-" * 40)
    df = fetch_training_data(engine)

    # 4. Prepare features
    X, y, feature_names = prepare_features(df)

    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 6. Train model
    print("\n" + "-" * 40)
    model = train_model(X_train, y_train)

    # 7. Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # 8. Feature importance
    get_feature_importance(model, feature_names)

    # 9. Save model
    print("\n" + "-" * 40)
    model_path = save_model(model, '/models/model.pkl')

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"   → Model: RandomForestRegressor")
    print(f"   → R² Score: {metrics['r2']:.4f}")
    print(f"   → RMSE: {metrics['rmse']:.4f}")
    print(f"   → Saved to: {model_path}")


if __name__ == "__main__":
    main()
