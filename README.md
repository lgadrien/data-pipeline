# Iris Data Pipeline

Pipeline de prédiction de longueur de sépale pour les fleurs Iris.

## Prérequis

- Docker
- Docker Compose

## Installation

```bash
git clone <repo-url>
cd datapipeline
docker compose up -d
```

## Accès

| Service  | URL                   |
| -------- | --------------------- |
| Frontend | http://localhost      |
| API      | http://localhost:8000 |

## Architecture

```
1. PostgreSQL      → Base de données
2. Preprocessing   → Normalise les données du CSV
3. Model Training  → Entraîne RandomForest (sepal_width → sepal_length)
4. API Flask       → Sert les prédictions
5. Frontend Nginx  → Interface utilisateur
```

## Endpoints API

### POST /predict

Prédit la longueur du sépale.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_width": 3.0}'
```

Réponse :

```json
{ "predicted_sepal_length": 5.6, "input": { "sepal_width": 3.0 } }
```

### GET /metrics

Retourne les performances du modèle.

```bash
curl http://localhost:8000/metrics
```

Réponse :

```json
{
  "r2_score": -0.6123,
  "rmse": 1.048,
  "mae": 0.7308,
  "model_type": "RandomForestRegressor",
  "features": ["sepal_width"],
  "target": "sepal_length"
}
```

## Commandes

```bash
docker compose up -d          # Démarrer
docker compose down           # Arrêter
docker compose up -d --build  # Rebuild
docker logs iris_api          # Logs API
```
