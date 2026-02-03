"""
================================================================================
                        API FASTAPI - SERVICE DE PRÉDICTION
================================================================================

DESCRIPTION:
    Ce fichier contient l'API REST qui permet d'utiliser le modèle entraîné
    pour faire des prédictions. Elle expose plusieurs endpoints HTTP.

RÔLE DANS L'ARCHITECTURE:
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   UTILISATEUR   │ ───▶ │   API FastAPI   │ ───▶ │     MODÈLE      │
    │  (navigateur)   │      │   (ce fichier)  │      │ (iris_model.joblib)
    └─────────────────┘      └─────────────────┘      └─────────────────┘

ENDPOINTS DISPONIBLES:
    GET  /           → Interface web (sert index.html)
    GET  /health     → Vérification de santé de l'API
    GET  /model/info → Informations sur le modèle chargé
    POST /predict    → Faire une prédiction
    GET  /docs       → Documentation Swagger auto-générée

TECHNOLOGIES:
    - FastAPI : Framework web moderne et performant pour créer des APIs
    - Pydantic : Validation automatique des données d'entrée/sortie
    - Uvicorn : Serveur ASGI pour exécuter FastAPI
    - Joblib : Chargement du modèle pré-entraîné

AUTEUR: Projet Data Pipeline - Epitech 2025-2026
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os                          # Accès aux variables d'environnement
from pathlib import Path           # Manipulation des chemins de fichiers
from fastapi import FastAPI, HTTPException  # Framework web + gestion des erreurs HTTP
from fastapi.staticfiles import StaticFiles # Servir des fichiers statiques (CSS, JS, images)
from fastapi.responses import FileResponse  # Renvoyer un fichier comme réponse HTTP
from pydantic import BaseModel     # Validation et sérialisation des données
import joblib                      # Chargement du modèle ML sauvegardé
import numpy as np                 # Calculs numériques (format attendu par le modèle)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Les valeurs sont récupérées depuis les variables d'environnement Docker.

# Chemin vers le modèle entraîné (sauvegardé par pipeline.py)
# Ce fichier est partagé entre le pipeline et l'API via un volume Docker
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/iris_model.joblib")

# URL du serveur MLflow (non utilisé directement ici, mais disponible pour référence)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Chemin vers le dossier contenant les fichiers statiques (HTML, CSS, JS)
# __file__ = chemin de ce fichier Python
# .parent = dossier parent (api/)
# / "static" = sous-dossier static
STATIC_DIR = Path(__file__).parent / "static"


# ==============================================================================
# INITIALISATION DE L'APPLICATION FASTAPI
# ==============================================================================

# Création de l'instance FastAPI avec métadonnées
# Ces informations apparaissent dans la documentation Swagger (/docs)
app = FastAPI(
    title="Iris Prediction API",
    description="API pour prédire la longueur des sépales (sepal_length) à partir de la largeur (sepal_width)",
    version="1.0.0"
)


# ==============================================================================
# MONTAGE DES FICHIERS STATIQUES
# ==============================================================================
# Permet de servir des fichiers CSS, JS, images depuis le dossier /static

if STATIC_DIR.exists():
    # Toutes les requêtes vers /static/xxx seront servies depuis le dossier static
    # Exemple: /static/style.css → api/static/style.css
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# VARIABLE GLOBALE POUR LE MODÈLE
# ==============================================================================
# Le modèle est chargé une seule fois au démarrage, puis réutilisé pour chaque requête

model = None  # Sera initialisé par load_model()


# ==============================================================================
# FONCTION DE CHARGEMENT DU MODÈLE
# ==============================================================================

def load_model():
    """
    Charge le modèle depuis le fichier .joblib sauvegardé par le pipeline.

    POURQUOI UN FICHIER .joblib ?
        - Format de sérialisation efficace pour les objets Python/NumPy
        - Plus rapide que pickle pour les gros objets avec des arrays
        - Standard dans scikit-learn pour sauvegarder les modèles

    FONCTIONNEMENT:
        1. Vérifie que le fichier existe
        2. Charge le modèle avec joblib.load()
        3. Stocke dans la variable globale 'model'

    Returns:
        Le modèle chargé, ou None si le chargement échoue
    """
    global model  # Permet de modifier la variable globale 'model'

    # Vérification de l'existence du fichier
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Modèle non trouvé: {MODEL_PATH}")
        return None

    try:
        # Chargement du modèle (désérialisation)
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None


# ==============================================================================
# SCHÉMAS PYDANTIC (VALIDATION DES DONNÉES)
# ==============================================================================
# Pydantic valide automatiquement les données entrantes et sortantes.
# Si les données ne correspondent pas au schéma, une erreur 422 est renvoyée.

class PredictionInput(BaseModel):
    """
    Schéma pour les données d'entrée de la prédiction.

    VALIDATION AUTOMATIQUE:
        - Vérifie que 'sepal_width' est présent
        - Vérifie que c'est un nombre décimal (float)
        - Convertit automatiquement les entiers en float

    Attributes:
        sepal_width: Largeur des sépales en cm (float)
    """
    sepal_width: float  # La seule entrée requise

    class Config:
        # Exemple affiché dans la documentation Swagger
        json_schema_extra = {
            "example": {
                "sepal_width": 3.5
            }
        }


class PredictionOutput(BaseModel):
    """
    Schéma pour la réponse de prédiction.

    Attributes:
        sepal_width: Valeur d'entrée (echo)
        predicted_sepal_length: Valeur prédite par le modèle
        model_path: Chemin du modèle utilisé
    """
    sepal_width: float
    predicted_sepal_length: float
    model_path: str


class HealthResponse(BaseModel):
    """
    Schéma pour la réponse du health check.

    Attributes:
        status: État de l'API ("healthy" ou "unhealthy")
        model_loaded: Indique si le modèle est chargé en mémoire
    """
    status: str
    model_loaded: bool


# ==============================================================================
# ÉVÉNEMENT DE DÉMARRAGE
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Fonction exécutée automatiquement au démarrage de l'API.

    POURQUOI CHARGER AU DÉMARRAGE ?
        - Le modèle est chargé une seule fois en mémoire
        - Évite de le recharger à chaque requête (performance)
        - Permet de détecter les erreurs dès le démarrage
    """
    load_model()


# ==============================================================================
# ROUTES (ENDPOINTS)
# ==============================================================================

# --- Route racine : Interface web ---
@app.get("/", tags=["Frontend"], include_in_schema=False)
async def root():
    """
    Sert l'interface web principale (index.html).

    FONCTIONNEMENT:
        1. Vérifie si le fichier index.html existe dans /static
        2. Si oui, le renvoie comme réponse HTTP
        3. Sinon, renvoie un JSON avec les liens utiles

    Note: include_in_schema=False masque cette route dans la doc Swagger
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        # FileResponse envoie le fichier avec le bon Content-Type
        return FileResponse(str(index_file))

    # Fallback si pas d'interface web
    return {
        "message": "Iris Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# --- Route de santé ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API.

    UTILITÉ:
        - Permet aux outils de monitoring de vérifier si l'API fonctionne
        - Docker peut utiliser cette route pour les health checks
        - Le frontend l'utilise pour afficher le badge de statut

    Returns:
        JSON avec le statut de l'API et du modèle
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None  # True si le modèle est chargé
    )


# --- Route principale : Prédiction ---
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Prédit la longueur des sépales à partir de la largeur.

    FLUX DE LA REQUÊTE:
        1. FastAPI reçoit le JSON {"sepal_width": 3.5}
        2. Pydantic valide les données (PredictionInput)
        3. On vérifie que le modèle est chargé
        4. On formate l'entrée en array NumPy 2D
        5. On appelle model.predict()
        6. On renvoie le résultat formaté (PredictionOutput)

    Args:
        input_data: Objet PredictionInput validé par Pydantic

    Returns:
        PredictionOutput avec la prédiction

    Raises:
        HTTPException 503: Si le modèle n'est pas disponible
        HTTPException 500: Si une erreur survient pendant la prédiction
    """
    global model

    # Vérifier que le modèle est chargé, sinon tenter de le recharger
    if model is None:
        model = load_model()
        if model is None:
            # Erreur 503 = Service Unavailable
            raise HTTPException(
                status_code=503,
                detail="Modèle non disponible. Le pipeline a-t-il été exécuté?"
            )

    try:
        # --- Préparation de l'entrée ---
        # Le modèle attend un array 2D : [[valeur]]
        # np.array([[3.5]]) → shape (1, 1)
        X = np.array([[input_data.sepal_width]])

        # --- Prédiction ---
        # model.predict() renvoie un array, on prend le premier élément [0]
        prediction = model.predict(X)[0]

        # --- Formatage de la réponse ---
        return PredictionOutput(
            sepal_width=input_data.sepal_width,
            predicted_sepal_length=round(prediction, 4),  # Arrondi à 4 décimales
            model_path=MODEL_PATH
        )

    except Exception as e:
        # Erreur 500 = Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


# --- Route d'information sur le modèle ---
@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Renvoie des informations sur le modèle chargé.

    INFORMATIONS RENVOYÉES:
        - Type de modèle (ex: RandomForestRegressor)
        - Chemin du fichier modèle
        - Hyperparamètres (n_estimators, max_depth)
        - Noms des features et target

    Returns:
        Dictionnaire avec les infos du modèle

    Raises:
        HTTPException 503: Si le modèle n'est pas chargé
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )

    return {
        "model_type": type(model).__name__,              # "RandomForestRegressor"
        "model_path": MODEL_PATH,                        # Chemin du fichier
        "n_estimators": getattr(model, "n_estimators", None),  # Nombre d'arbres
        "max_depth": getattr(model, "max_depth", None),        # Profondeur max
        "feature_names": ["sepal_width"],                # Variable d'entrée
        "target_name": "sepal_length"                    # Variable de sortie
    }


# ==============================================================================
# NOTE: POUR LANCER L'API MANUELLEMENT (hors Docker)
# ==============================================================================
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
#
# - app:app = fichier app.py, variable app (l'instance FastAPI)
# - --reload = Rechargement automatique en cas de modification du code
# - --host 0.0.0.0 = Écoute sur toutes les interfaces réseau
# - --port 8000 = Port d'écoute
# ==============================================================================
