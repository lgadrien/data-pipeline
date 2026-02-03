"""
API FastAPI - Service de prédiction
====================================
Expose une route /predict pour prédire sepal_length à partir de sepal_width
Charge le modèle depuis le volume partagé (sauvegardé par le pipeline)
Inclut une interface web pour interagir avec l'API
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/iris_model.joblib")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
STATIC_DIR = Path(__file__).parent / "static"

# Initialisation FastAPI
app = FastAPI(
    title="Iris Prediction API",
    description="API pour prédire la longueur des sépales (sepal_length) à partir de la largeur (sepal_width)",
    version="1.0.0"
)

# Montage des fichiers statiques
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Chargement du modèle au démarrage
model = None


def load_model():
    """Charge le modèle depuis le fichier joblib"""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Modèle non trouvé: {MODEL_PATH}")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None


# Schémas Pydantic
class PredictionInput(BaseModel):
    sepal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_width": 3.5
            }
        }


class PredictionOutput(BaseModel):
    sepal_width: float
    predicted_sepal_length: float
    model_path: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Routes
@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    load_model()


@app.get("/", tags=["Frontend"], include_in_schema=False)
async def root():
    """Sert l'interface web principale"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Iris Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifie l'état de santé de l'API"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Prédit la longueur des sépales (sepal_length)
    à partir de la largeur (sepal_width)
    """
    global model

    # Recharger le modèle si nécessaire
    if model is None:
        model = load_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modèle non disponible. Le pipeline a-t-il été exécuté?"
            )

    try:
        # Prédiction
        X = np.array([[input_data.sepal_width]])
        prediction = model.predict(X)[0]

        return PredictionOutput(
            sepal_width=input_data.sepal_width,
            predicted_sepal_length=round(prediction, 4),
            model_path=MODEL_PATH
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )

    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None),
        "feature_names": ["sepal_width"],
        "target_name": "sepal_length"
    }
