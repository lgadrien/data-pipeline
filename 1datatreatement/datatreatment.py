"""
Data Preprocessing Script for Iris Dataset
==========================================
This script performs the following ETL operations:
1. Load environment variables for secure database access
2. Wait for PostgreSQL to be ready (wait strategy)
3. Load and clean the iris.csv data
4. Normalize numerical features with StandardScaler
5. Store processed data in PostgreSQL
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT
# =============================================================================

def get_db_config():
    """Load database configuration from environment variables."""
    return {
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'host': os.getenv('POSTGRES_HOST', 'db'),  # 'db' is the service name in docker-compose
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'datapipeline')
    }


def build_connection_url(config):
    """Build PostgreSQL connection URL from config."""
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"


def wait_for_database(engine, max_retries=10, retry_interval=5):
    """
    Wait Strategy: Attempt to connect to the database with retries.
    This handles the case where the script starts before PostgreSQL is ready.
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
# 2. DATA LOADING (Pandas)
# =============================================================================

def load_csv(filepath):
    """Load the iris CSV file."""
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   → Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def normalize_column_names(df):
    """
    Clean and normalize column names to snake_case.
    Handles: spaces, dots, and inconsistent casing.
    """
    print("🔧 Normalizing column names...")

    # Store original names for comparison
    original_names = df.columns.tolist()

    # Normalize: lowercase, replace spaces and dots with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_', regex=False)
        .str.replace('.', '_', regex=False)
        .str.replace('__', '_', regex=False)  # Remove double underscores
    )

    # Log changes
    for old, new in zip(original_names, df.columns):
        if old != new:
            print(f"   → Renamed: '{old}' → '{new}'")

    print(f"   → Final columns: {df.columns.tolist()}")
    return df


# =============================================================================
# 3. CLEANING & NORMALIZATION
# =============================================================================

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    print("🧹 Checking for duplicates...")

    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(f"   → Removed {removed_count} duplicate rows")
    else:
        print("   → No duplicates found")

    return df


def normalize_features(df, target_column='sepal_length'):
    """
    Normalize numerical features using StandardScaler.
    The target column (sepal_length) is NOT normalized to keep predictions readable.
    """
    print("📊 Normalizing numerical features...")

    # Identify numerical columns (excluding target and species)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target column from normalization
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    print(f"   → Features to normalize: {numerical_cols}")
    print(f"   → Target column (not normalized): {target_column}")

    # Apply StandardScaler
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Log normalization stats
    print("   → Normalization complete (mean=0, std=1)")
    for col in numerical_cols:
        print(f"      {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    return df


# =============================================================================
# 4. STORAGE (SQLAlchemy)
# =============================================================================

def save_to_database(df, engine, table_name='iris_data'):
    """Save the processed DataFrame to PostgreSQL."""
    print(f"💾 Saving data to PostgreSQL table '{table_name}'...")

    df.to_sql(
        table_name,
        engine,
        if_exists='replace',  # Replace table if it exists
        index=False           # Don't save DataFrame index
    )

    # Verify the save
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()

    print(f"✅ Successfully saved {count} rows to '{table_name}'")
    return count


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main ETL pipeline execution."""
    print("=" * 60)
    print("🚀 IRIS DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Configuration
    config = get_db_config()
    print(f"\n📌 Database: {config['host']}:{config['port']}/{config['database']}")

    connection_url = build_connection_url(config)
    engine = create_engine(connection_url)

    # 2. Wait for database
    wait_for_database(engine)

    # 3. Load data
    print("\n" + "-" * 40)
    df = load_csv("/data/iris.csv")

    # 4. Clean column names
    df = normalize_column_names(df)

    # 5. Remove duplicates
    df = remove_duplicates(df)

    # 6. Normalize features
    df = normalize_features(df, target_column='sepal_length')

    # 7. Save to database
    print("\n" + "-" * 40)
    save_to_database(df, engine, table_name='iris_data')

    # Summary
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"   → Rows processed: {len(df)}")
    print(f"   → Columns: {df.columns.tolist()}")
    print(f"   → Table: iris_data")


if __name__ == "__main__":
    main()
