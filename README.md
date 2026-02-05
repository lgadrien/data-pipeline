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
3. Model Training  → Entraîne RandomForest + Polynomial Features (degree=3) (sepal_width → sepal_length)
4. API Flask       → Sert les prédictions
5. Frontend Nginx  → Interface utilisateur
```

## Endpoints API

### GET /health

Vérifie l'état de l'API et du modèle.

```bash
curl http://localhost:8000/health
```

Réponse :

```json
{ "status": "healthy" }
```

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
  "r2_score": -0.6088,
  "rmse": 1.0468,
  "mae": 0.7268,
  "train_samples": 119,
  "test_samples": 30,
  "model_type": "Pipeline(PolynomialFeatures + RandomForest)",
  "polynomial_degree": 3,
  "n_estimators": 200,
  "input_feature": "sepal_width",
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
