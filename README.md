# Projet MLOps – Prédiction du revenu des adultes

## Description

Ce projet illustre un pipeline complet de Machine Learning en MLOps, comprenant :

* Préparation et nettoyage des données.
* Entraînement d’un modèle de classification pour prédire si le revenu d’un individu dépasse 50K$/an.
* Évaluation des performances avec des métriques standard (accuracy, F1-score, etc.).
* API FastAPI pour les prédictions.
* Interface web Streamlit pour l’exploration des données et la prédiction individuelle ou en batch.
* Profiling automatisé du dataset avec un rapport HTML interactif.
* Tracking et versioning avec MLflow.
* CI/CD avec tests automatiques et déploiement continu via Render.

Le projet utilise le dataset Adult Census Income provenant de l’UCI Machine Learning Repository :
[Adult Dataset – UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)

## Structure du projet
```
MLOPS_TP1/
├── data/ # Données brutes et CSV
│   └── adult/
├── src/ # Code source
│   ├── api.py # API FastAPI pour les prédictions
│   ├── app.py # Streamlit pour interface utilisateur
│   ├── training.py # Entraînement du modèle
│   ├── analysis.py # Analyse exploratoire
│   └── artifacts/ # Modèles, rapports, métriques sauvegardés
├── docker_api/ # Dockerfile pour l’API
│   └── Dockerfile
├── docker_front/ # Dockerfile pour l’interface Streamlit
│   └── Dockerfile
├── mlflow.db # Base de données MLflow
├── mlruns/ # Historique des runs MLflow
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier
```
## Liens vers les applications déployées

* API FastAPI : [https://adult-ml-api.onrender.com](https://adult-ml-api.onrender.com)
* Interface Streamlit : [https://adult-ml-app.onrender.com](https://adult-ml-app.onrender.com)

## Commandes utiles

### Python & Virtual Environment
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### Lancer l’API FastAPI

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints :

* GET /health → état du service
* GET /metadata → métadonnées du modèle
* POST /predict → prédictions (single ou batch)

### Lancer l’application Streamlit

streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0

### Docker
```
**API**
docker build -f docker_api/Dockerfile -t mlops-api .
docker run -p 8000:8000 mlops-api

**Streamlit**
docker build -f docker_front/Dockerfile -t mlops-streamlit .
docker run -p 8501:8501 mlops-streamlit

**Vérifier les conteneurs**
docker ps
```
### MLflow

mlflow ui --backend-store-uri sqlite:///mlflow.db

Interface MLflow disponible sur [http://localhost:5000](http://localhost:5000).

## Tests

Le projet comprend des tests pour :

* L’API (test_api.py)
* Les fonctions d’inférence (test_inference.py)
* L’entraînement du modèle (test_training.py)

Exécuter tous les tests :

pytest src/

## CI/CD

* CI : GitHub Actions build et teste automatiquement :

  * Build des images Docker pour l’API et Streamlit.
  * Exécution de tous les tests pytest.
## CD
* Deploiement on commit avec render