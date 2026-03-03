# Projet MLOps – Prédiction du revenu des adultes

## Description

Ce projet illustre un **pipeline complet de Machine Learning en MLOps**, comprenant :

- Préparation et nettoyage des données.
- Entraînement d’un modèle de classification pour prédire si le revenu d’un individu dépasse 50K$/an.
- Évaluation des performances avec des métriques standard (accuracy, F1-score, etc.).
- API FastAPI pour les prédictions.
- Interface web Streamlit pour l’exploration des données et la prédiction individuelle ou en batch.
- Profiling automatisé du dataset avec un rapport HTML interactif.

Le projet utilise le dataset **Adult Census Income** provenant de l’UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/2/adult
---

## Structure du projet
```
MLOPS_TP1/
├── data/ # Données brutes et CSV
│ └── adult/
├── src/ # Code source
│ ├── api.py # API FastAPI pour les prédictions
│ ├── app.py # Streamlit pour interface utilisateur
│ ├── training.py # Entraînement du modèle
│ ├── analysis.py # Analyse exploratoire
│ └── artifacts/ # Modèles, rapports, métriques sauvegardés
├── Dockerfile # Conteneurisation
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier
```
## Liste des commandes a portentiellement utiliser

## Venv
source .venv/bin/activate
## Requirement
pip freeze > requirements.txt
pip install -r requirements.txt
## API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
## Application
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
## A faire
Docker (permission admin)