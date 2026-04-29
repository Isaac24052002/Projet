# Telco Churn API (XGBoost)

API FastAPI prête à déployer avec interface web moderne pour scorer le risque de churn à partir du modèle `churn_xgb_pipeline.joblib`.

## Structure
- `app/main.py` : backend FastAPI (API JSON + interface web)
- `templates/index.html` : interface utilisateur
- `static/css/styles.css` : design responsive
- `requirements.txt` : dépendances
- `Dockerfile` : image de déploiement
- `Procfile` : lancement sur plateformes type Render
- `run.sh` : démarrage local rapide

## Modèle attendu
Place le fichier modèle à la racine du projet:
- `churn_xgb_pipeline.joblib`

Ou définis une variable d'environnement:
- `MODEL_PATH=/chemin/vers/churn_xgb_pipeline.joblib`

## Lancer en local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Interface:
- `http://localhost:8000/`

Documentation API:
- `http://localhost:8000/docs`

## Endpoint principal
### `POST /api/predict`
Payload exemple:
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

Réponse:
```json
{
  "prediction": 1,
  "probability": 0.8231,
  "risk_level": "High"
}
```

## Déployer avec Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 -e MODEL_PATH=/app/churn_xgb_pipeline.joblib churn-api
```

## Vérification santé
- `GET /health`
