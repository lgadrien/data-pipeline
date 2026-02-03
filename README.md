# 🌸 Iris Data Pipeline

Pipeline de données complet avec **4 services Docker** indépendants pour l'entraînement et le déploiement d'un modèle de Machine Learning sur le dataset Iris.

![Interface Web](https://img.shields.io/badge/Frontend-Modern%20UI-7c3aed?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?style=for-the-badge&logo=mlflow)

## 📋 Description

Ce projet implémente un pipeline ETL (Extract, Transform, Load) complet qui :

1. **Charge** les données Iris depuis un fichier CSV
2. **Stocke** les données dans PostgreSQL
3. **Entraîne** un modèle de régression (RandomForest) pour prédire la longueur des sépales
4. **Expose** une API REST pour faire des prédictions
5. **Propose** une interface web moderne pour interagir avec le modèle

## ✨ Fonctionnalités

- 🎨 **Interface Web Moderne** - Design dark mode avec animations et effets glassmorphism
- 🔮 **Prédiction en temps réel** - API REST performante avec FastAPI
- 📊 **Tracking MLOps** - Suivi des expériences avec MLflow
- 🐘 **Stockage PostgreSQL** - Persistance des données
- 🐳 **Entièrement Dockerisé** - Déploiement simple avec Docker Compose

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  PostgreSQL  │    │    MLflow    │    │   FastAPI    │       │
│  │    (db)      │    │   (mlflow)   │    │ (api + web)  │       │
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
│   ├── app.py           # API FastAPI
│   └── static/
│       └── index.html   # Interface web moderne
├── data/
│   └── iris.csv         # Dataset Iris
└── documentation/       # Documentation du projet
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

## 🌐 Interfaces Web

| Service              | URL                        | Description                         |
| -------------------- | -------------------------- | ----------------------------------- |
| **🎯 Interface Web** | http://localhost:8000      | Interface de prédiction interactive |
| **📄 API Swagger**   | http://localhost:8000/docs | Documentation interactive de l'API  |
| **📊 MLflow UI**     | http://localhost:5001      | Suivi des expériences ML            |

## 🎨 Interface Web

L'application dispose d'une interface web moderne et élégante pour faire des prédictions :

### Fonctionnalités de l'interface

- 🎚️ **Slider interactif** - Ajustez facilement la valeur de sepal_width
- 🔮 **Prédiction instantanée** - Résultats en temps réel
- 📊 **Badge de statut** - Vérifie si l'API et le modèle sont prêts
- 🌙 **Design Dark Mode** - Interface moderne avec effets visuels
- ✨ **Animations fluides** - Particules, transitions, hover effects
- 📱 **Responsive** - Compatible mobile et desktop

### Comment l'utiliser

1. Ouvrir http://localhost:8000 dans votre navigateur
2. Ajuster la valeur de `sepal_width` avec le slider ou le champ texte
3. Cliquer sur **"🔮 Prédire la longueur"**
4. Le résultat s'affiche instantanément avec la longueur prédite

## 🔮 Faire une prédiction

### Option 1 : Via l'interface Web (Recommandé)

1. Ouvrir http://localhost:8000 dans votre navigateur
2. Entrer une valeur pour `sepal_width` (entre 2.0 et 4.5 cm)
3. Cliquer sur **"Prédire la longueur"**
4. Voir le résultat affiché avec une animation

### Option 2 : Via curl (Terminal)

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

### Option 3 : Via Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"sepal_width": 3.5}
)

print(response.json())
# {'sepal_width': 3.5, 'predicted_sepal_length': 5.0815, ...}
```

### Option 4 : Via l'interface Swagger

1. Ouvrir http://localhost:8000/docs dans votre navigateur
2. Cliquer sur **POST /predict**
3. Cliquer sur **Try it out**
4. Entrer une valeur pour `sepal_width` (ex: 3.5)
5. Cliquer sur **Execute**

## 📊 Endpoints de l'API

| Méthode | Endpoint      | Description                 |
| ------- | ------------- | --------------------------- |
| `GET`   | `/`           | Interface web de prédiction |
| `GET`   | `/health`     | Statut de santé de l'API    |
| `GET`   | `/model/info` | Informations sur le modèle  |
| `POST`  | `/predict`    | Faire une prédiction        |
| `GET`   | `/docs`       | Documentation Swagger       |

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

| Caractéristique | Valeur                              |
| --------------- | ----------------------------------- |
| **Type**        | RandomForestRegressor               |
| **Feature**     | sepal_width (largeur des sépales)   |
| **Target**      | sepal_length (longueur des sépales) |
| **Estimateurs** | 100 arbres                          |
| **Max Depth**   | 5                                   |

### Métriques

| Métrique     | Valeur   |
| ------------ | -------- |
| **RMSE**     | ~0.85    |
| **MAE**      | ~0.65    |
| **R² Score** | Variable |

## 🛠️ Technologies utilisées

| Catégorie     | Technologies                      |
| ------------- | --------------------------------- |
| **Backend**   | Python, FastAPI, SQLAlchemy       |
| **ML**        | Scikit-learn, RandomForest        |
| **MLOps**     | MLflow                            |
| **Database**  | PostgreSQL                        |
| **Frontend**  | HTML5, CSS3, JavaScript (Vanilla) |
| **Container** | Docker, Docker Compose            |

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

### L'interface web ne s'affiche pas

```bash
# Vérifier que les fichiers statiques sont bien montés
docker exec api_service ls -la /app/api/static/

# Reconstruire l'image
docker compose up -d --build api
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
