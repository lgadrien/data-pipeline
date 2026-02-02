# 🌸 Iris Data Pipeline

Pipeline de données complet avec **4 services Docker** indépendants pour l'entraînement et le déploiement d'un modèle de Machine Learning sur le dataset Iris.

## 📋 Description

Ce projet implémente un pipeline ETL (Extract, Transform, Load) qui :

1. **Charge** les données Iris depuis un fichier CSV
2. **Stocke** les données dans PostgreSQL
3. **Entraîne** un modèle de régression (RandomForest) pour prédire la longueur des sépales
4. **Expose** une API REST pour faire des prédictions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  PostgreSQL  │    │    MLflow    │    │   FastAPI    │       │
│  │    (db)      │    │   (mlflow)   │    │    (api)     │       │
│  │  Port: 5432  │    │  Port: 5001  │    │  Port: 8000  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ▲                   ▲                   ▲                │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────┴────────┐                          │
│                    │    Pipeline     │                          │
│                    │  (pipeline_etl) │                          │
│                    │   ETL + Train   │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Structure du projet

```
datapipeline/
├── docker-compose.yml   # Orchestration des 4 services
├── Dockerfile           # Image Python pour pipeline et API
├── requirements.txt     # Dépendances Python
├── pipeline.py          # Script ETL + Training
├── README.md            # Ce fichier
├── api/
│   └── app.py           # API FastAPI
└── data/
    └── iris.csv         # Dataset Iris
```

## 🚀 Démarrage rapide

### Prérequis

- [Docker](https://www.docker.com/get-started) installé
- [Docker Compose](https://docs.docker.com/compose/) installé

### Lancer le projet

```bash
# Cloner le projet (si nécessaire)
cd datapipeline

# Lancer tous les services
docker compose up -d --build

# Vérifier que tout fonctionne
docker compose ps
```

### Résultat attendu

```
NAME              IMAGE                           STATUS                    PORTS
api_service       datapipeline-api                Up                        0.0.0.0:8000->8000
mlflow_tracking   ghcr.io/mlflow/mlflow:v2.10.0   Up                        0.0.0.0:5001->5000
postgres_db       postgres:15-alpine              Up (healthy)              0.0.0.0:5432->5432
```

## 🔮 Faire une prédiction

### Option 1 : Via curl (Terminal)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_width": 3.5}'
```

**Réponse :**

```json
{
  "sepal_width": 3.5,
  "predicted_sepal_length": 5.0815,
  "model_path": "/app/models/iris_model.joblib"
}
```

### Option 2 : Via Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"sepal_width": 3.5}
)

print(response.json())
# {'sepal_width': 3.5, 'predicted_sepal_length': 5.0815, ...}
```

### Option 3 : Via l'interface Swagger

1. Ouvrir http://localhost:8000/docs dans votre navigateur
2. Cliquer sur **POST /predict**
3. Cliquer sur **Try it out**
4. Entrer une valeur pour `sepal_width` (ex: 3.5)
5. Cliquer sur **Execute**

## 🌐 Interfaces Web

| Service         | URL                        | Description                        |
| --------------- | -------------------------- | ---------------------------------- |
| **API Swagger** | http://localhost:8000/docs | Documentation interactive de l'API |
| **MLflow UI**   | http://localhost:5001      | Suivi des expériences ML           |

## 📊 Endpoints de l'API

| Méthode | Endpoint      | Description                |
| ------- | ------------- | -------------------------- |
| `GET`   | `/`           | Page d'accueil             |
| `GET`   | `/health`     | Statut de santé de l'API   |
| `GET`   | `/model/info` | Informations sur le modèle |
| `POST`  | `/predict`    | Faire une prédiction       |

### Exemple de requête `/predict`

**Request:**

```json
{
  "sepal_width": 3.5
}
```

**Response:**

```json
{
  "sepal_width": 3.5,
  "predicted_sepal_length": 5.0815,
  "model_path": "/app/models/iris_model.joblib"
}
```

## 🔧 Commandes utiles

```bash
# Voir les logs de tous les services
docker compose logs -f

# Voir les logs d'un service spécifique
docker compose logs -f api
docker compose logs -f pipeline

# Relancer uniquement le pipeline (ré-entraîner le modèle)
docker compose up pipeline

# Arrêter tous les services
docker compose down

# Arrêter et supprimer les données (volumes)
docker compose down -v

# Reconstruire les images après modification du code
docker compose up -d --build
```

## 🗄️ Accéder à PostgreSQL

```bash
# Se connecter à la base de données
docker exec -it postgres_db psql -U admin -d datapipeline

# Voir les données
SELECT * FROM iris_data LIMIT 5;

# Compter les lignes
SELECT COUNT(*) FROM iris_data;

# Quitter
\q
```

## 📈 Le modèle

- **Type** : RandomForestRegressor
- **Feature** : `sepal_width` (largeur des sépales)
- **Target** : `sepal_length` (longueur des sépales)
- **Métriques** :
  - RMSE : ~0.85
  - MAE : ~0.65

## 🐛 Dépannage

### Le pipeline ne démarre pas

```bash
# Vérifier les logs du pipeline
docker compose logs pipeline

# S'assurer que PostgreSQL est prêt
docker compose logs db
```

### L'API ne répond pas

```bash
# Vérifier que l'API est bien démarrée
docker compose ps

# Vérifier les logs de l'API
docker compose logs api
```

### Réinitialiser complètement

```bash
# Tout supprimer et recommencer
docker compose down -v
docker compose up -d --build
```

## 📝 Auteur

Projet réalisé dans le cadre du module **Data Pipeline** - Epitech 2025-2026

## 📄 License

MIT
