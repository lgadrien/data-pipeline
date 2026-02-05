# Dossier Technique - Iris Data Pipeline

## I. Le schéma d'architecture du pipeline

Le projet Iris Data Pipeline est une architecture microservices orchestrée par Docker Compose, composée de 4 services indépendants qui communiquent via un réseau Docker interne.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Docker Compose Network                                 │
│                          (pipeline_network - bridge)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────┐                                                          │
│  │      iris.csv      │  ← Données source (150 échantillons de fleurs Iris)     │
│  └─────────┬──────────┘                                                          │
│            │                                                                     │
│            ▼                                                                     │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐     │
│  │     PostgreSQL     │    │       MLflow       │    │      FastAPI       │     │
│  │    (postgres_db)   │    │  (mlflow_tracking) │    │   (api_service)    │     │
│  │                    │    │                    │    │                    │     │
│  │  • Port: 5432      │    │  • Port: 5001      │    │  • Port: 8000      │     │
│  │  • Image: postgres │    │  • Tracking server │    │  • REST API        │     │
│  │    :15-alpine      │    │  • Stockage        │    │  • Interface web   │     │
│  │  • Stockage des    │    │    expériences     │    │  • Prédictions     │     │
│  │    données iris    │    │  • Artefacts ML    │    │                    │     │
│  └─────────▲──────────┘    └─────────▲──────────┘    └─────────▲──────────┘     │
│            │                         │                         │                 │
│            │        ┌────────────────┴────────────────┐        │                 │
│            │        │                                 │        │                 │
│            └────────┤        Pipeline ETL             ├────────┘                 │
│                     │      (pipeline_etl)             │                          │
│                     │                                 │                          │
│                     │  1. EXTRACT : Lecture CSV       │                          │
│                     │  2. LOAD : Insertion PostgreSQL │                          │
│                     │  3. TRAIN : Entraînement ML     │                          │
│                     │                                 │                          │
│                     └─────────────────────────────────┘                          │
│                                     │                                            │
│                                     ▼                                            │
│                     ┌─────────────────────────────────┐                          │
│                     │      iris_model.joblib          │                          │
│                     │   (Volume partagé: model_artifacts)                        │
│                     └─────────────────────────────────┘                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │         UTILISATEUR             │
                    │   • Interface web (port 8000)   │
                    │   • API REST                    │
                    │   • Swagger UI (/docs)          │
                    └─────────────────────────────────┘
```

**Flux de données :**

1. Le fichier `iris.csv` est lu par le pipeline ETL
2. Les données sont insérées dans PostgreSQL (table `iris_data`)
3. Le modèle RandomForest est entraîné sur ces données
4. Les métriques sont enregistrées dans MLflow
5. Le modèle est sauvegardé dans un volume partagé
6. L'API FastAPI charge le modèle et expose les prédictions
7. L'utilisateur interagit via l'interface web ou l'API REST

---

## II. Les choix techniques

### a. Outils

| Outil              | Version   | Rôle                                                                        |
| ------------------ | --------- | --------------------------------------------------------------------------- |
| **Docker**         | Latest    | Conteneurisation des services pour portabilité et isolation                 |
| **Docker Compose** | Latest    | Orchestration multi-conteneurs, gestion des dépendances entre services      |
| **PostgreSQL**     | 15-alpine | Base de données relationnelle pour la persistance des données Iris          |
| **MLflow**         | 2.10.0    | Plateforme MLOps pour le tracking des expériences et la gestion des modèles |
| **Uvicorn**        | 0.27.0    | Serveur ASGI haute performance pour exécuter FastAPI                        |

**Justification des choix :**

- **Docker** : Garantit un environnement reproductible sur toute machine
- **PostgreSQL** : Base robuste, performante, adaptée aux données structurées
- **MLflow** : Standard de l'industrie pour le suivi des expériences ML

### b. Librairies

| Librairie           | Version               | Utilisation                                             |
| ------------------- | --------------------- | ------------------------------------------------------- |
| **pandas**          | 2.0.3                 | Manipulation et transformation des données (DataFrames) |
| **numpy**           | 1.24.3                | Calculs numériques et manipulation des arrays           |
| **scikit-learn**    | 1.3.2                 | Algorithmes de Machine Learning (RandomForestRegressor) |
| **sqlalchemy**      | 2.0.23                | ORM pour la connexion et les requêtes PostgreSQL        |
| **psycopg2-binary** | 2.9.9                 | Driver PostgreSQL pour Python                           |
| **fastapi**         | 0.109.0               | Framework web moderne pour créer l'API REST             |
| **pydantic**        | (inclus avec FastAPI) | Validation et sérialisation des données                 |
| **joblib**          | 1.3.2                 | Sérialisation/désérialisation du modèle ML              |
| **mlflow**          | 2.10.0                | SDK Python pour le tracking MLflow                      |

**Justification des choix :**

- **FastAPI** : Performances élevées, documentation auto-générée (Swagger), validation automatique
- **Scikit-learn** : Bibliothèque ML mature, bien documentée, parfaite pour les projets de taille moyenne
- **Pandas** : Standard de facto pour la manipulation de données en Python

### c. Design

**Architecture Microservices :**

- Séparation des responsabilités (ETL, stockage, ML, API)
- Scalabilité indépendante de chaque service
- Facilité de maintenance et de déploiement

**Interface utilisateur :**

- Design **Dark Mode** moderne et élégant
- Effets **Glassmorphism** pour un rendu premium
- Animations fluides (particules, transitions, hover effects)
- Interface **responsive** (compatible mobile et desktop)
- Slider interactif pour la saisie des valeurs

**API Design :**

- Architecture **REST** avec endpoints bien définis
- Validation des entrées via **Pydantic**
- Documentation **Swagger** auto-générée (`/docs`)
- Health check pour le monitoring (`/health`)

---

## III. Les performances du modèle

### a. Le modèle (RandomForest)

**Type de problème :** Régression (prédire une valeur continue)

**Objectif :** Prédire `sepal_length` (longueur des sépales) à partir de `sepal_width` (largeur des sépales)

**Algorithme choisi :** `RandomForestRegressor`

| Hyperparamètre      | Valeur | Description                                                     |
| ------------------- | ------ | --------------------------------------------------------------- |
| `n_estimators`      | 100    | Nombre d'arbres de décision dans la forêt                       |
| `max_depth`         | 5      | Profondeur maximale de chaque arbre (évite le surapprentissage) |
| `min_samples_split` | 2      | Nombre minimum d'échantillons pour diviser un nœud              |
| `min_samples_leaf`  | 1      | Nombre minimum d'échantillons dans une feuille                  |
| `random_state`      | 42     | Graine aléatoire pour la reproductibilité                       |

**Pourquoi RandomForest ?**

- Robuste face au bruit dans les données
- Nécessite peu de prétraitement
- Bon compromis biais/variance
- Adapté aux petits datasets (150 échantillons)
- Évite le surapprentissage grâce à l'ensemble d'arbres

**Répartition des données :**

- 80% pour l'entraînement (120 échantillons)
- 20% pour le test (30 échantillons)

### b. Les performances

| Métrique     | Valeur   | Interprétation                                              |
| ------------ | -------- | ----------------------------------------------------------- |
| **RMSE**     | ~0.85    | Erreur quadratique moyenne - pénalise les grandes erreurs   |
| **MAE**      | ~0.65    | Erreur absolue moyenne - plus intuitive (~0.65 cm d'erreur) |
| **R² Score** | Variable | Coefficient de détermination - qualité de l'ajustement      |

**Analyse des performances :**

- Le modèle prédit avec une erreur moyenne d'environ **0.65 cm**
- La corrélation entre `sepal_width` et `sepal_length` est modérée
- Les performances sont limitées par l'utilisation d'une seule feature
- L'ajout de features supplémentaires (petal_width, petal_length) améliorerait significativement les résultats

**Suivi des expériences :**

- Toutes les métriques sont trackées dans **MLflow**
- Accessible via l'interface MLflow UI sur le port 5001
- Historique complet des runs et comparaison possible

---

## IV. Analyse critique du projet et pistes d'amélioration

### a. Points forts du projet

1. **Architecture modulaire et scalable**
   - Séparation claire des responsabilités (ETL, BDD, ML, API)
   - Chaque service peut être mis à l'échelle indépendamment
   - Facilité de maintenance et de mise à jour

2. **Reproductibilité**
   - Docker garantit le même environnement partout
   - MLflow track les expériences et permet de recréer n'importe quel modèle
   - `random_state` fixé pour des résultats reproductibles

3. **Documentation complète**
   - Code source entièrement commenté en français
   - README détaillé avec instructions d'utilisation
   - Documentation Swagger auto-générée pour l'API

4. **Interface utilisateur moderne**
   - Design soigné avec effets visuels premium
   - Expérience utilisateur intuitive
   - Responsive design

5. **Bonnes pratiques DevOps**
   - Health checks pour la surveillance des services
   - Gestion propre des dépendances entre services
   - Volumes persistants pour les données

6. **Robustesse**
   - Gestion des erreurs à chaque étape
   - Système de retry pour les connexions aux services
   - Validation des entrées utilisateur

### b. Faiblesses du projet

1. **Modèle trop simpliste**
   - Utilisation d'une seule feature (`sepal_width`)
   - Performances limitées (~R² modéré)
   - Pas de feature engineering

2. **Pas de pipeline CI/CD**
   - Déploiement manuel
   - Pas de tests automatisés
   - Pas d'intégration continue

3. **Sécurité basique**
   - Credentials en dur dans le docker-compose
   - Pas d'authentification sur l'API
   - Pas de HTTPS

4. **Scalabilité limitée**
   - Base SQLite pour MLflow (non adapté à la production)
   - Pas de load balancer pour l'API
   - Pas de cache pour les prédictions

5. **Monitoring insuffisant**
   - Pas de système de logging centralisé
   - Pas d'alertes automatiques
   - Métriques système non collectées

**Pistes d'amélioration :**

- Ajouter des features supplémentaires au modèle
- Implémenter un pipeline CI/CD avec GitHub Actions
- Ajouter l'authentification JWT sur l'API
- Migrer MLflow vers PostgreSQL backend
- Implémenter un système de monitoring (Prometheus/Grafana)
- Ajouter des tests unitaires et d'intégration

---

## V. La répartition du travail

### a. Adrien

- **Rôle principal :** Lead technique & DevOps
- **Contributions :**
  - Architecture globale du projet
  - Configuration Docker et Docker Compose
  - Mise en place de l'orchestration des services
  - Intégration MLflow pour le tracking
  - Rédaction de la documentation technique

### b. Kamel

- **Rôle principal :** Data Engineer
- **Contributions :**
  - Développement du pipeline ETL (`pipeline.py`)
  - Extraction et transformation des données
  - Intégration PostgreSQL
  - Gestion de la persistance des données
  - Tests du flux de données

### c. Sacha

- **Rôle principal :** Machine Learning Engineer
- **Contributions :**
  - Choix et implémentation du modèle RandomForest
  - Entraînement et évaluation du modèle
  - Optimisation des hyperparamètres
  - Sérialisation du modèle avec joblib
  - Analyse des métriques de performance

### d. Josh

- **Rôle principal :** Backend & Frontend Developer
- **Contributions :**
  - Développement de l'API FastAPI (`app.py`)
  - Création des endpoints REST
  - Développement de l'interface web (`index.html`)
  - Design UI/UX avec effets visuels modernes
  - Intégration frontend-API

---

_Document rédigé dans le cadre du module Data Pipeline - Epitech 2025-2026_
