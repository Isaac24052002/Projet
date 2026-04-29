import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import _column_transformer
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "churn_xgb_pipeline.joblib"


class ChurnFeatures(BaseModel):
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(default=12, ge=0)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "DSL"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(default=70.0, ge=0)
    TotalCharges: float = Field(default=840.0, ge=0)


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            BASE_DIR / DEFAULT_MODEL_NAME,
            Path.cwd() / DEFAULT_MODEL_NAME,
            Path.home() / "Téléchargements" / DEFAULT_MODEL_NAME,
            Path.home() / "Downloads" / DEFAULT_MODEL_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_sklearn_pickle_compat() -> None:
    # Compatibility shim for models serialized with older sklearn internals.
    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _column_transformer._RemainderColsList = _RemainderColsList


MODEL_PATH = _resolve_model_path()
if MODEL_PATH.exists():
    _ensure_sklearn_pickle_compat()
    model = joblib.load(MODEL_PATH)
else:
    model = None

app = FastAPI(title="Telco Churn API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _predict_score(features: ChurnFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Add churn_xgb_pipeline.joblib and restart the app.",
        )

    row = pd.DataFrame([features.model_dump()])
    row_for_model = row

    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        if not set(expected).issubset(set(row.columns)):
            # Some saved pipelines/models expect one-hot encoded columns as input.
            # Build a compatible row by dummy-encoding and reindexing to expected features.
            encoded = pd.get_dummies(row, drop_first=False)
            row_for_model = encoded.reindex(columns=expected, fill_value=0)
        else:
            row_for_model = row.reindex(columns=expected, fill_value=0)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row_for_model)[0, 1])
    else:
        raw = float(model.predict(row_for_model)[0])
        proba = max(0.0, min(1.0, raw))

    prediction = int(proba >= 0.5)
    return prediction, proba


def _risk_label(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/predict")
def predict_api(features: ChurnFeatures):
    prediction, probability = _predict_score(features)
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": _risk_label(probability),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "model_loaded": model is not None,
            "model_path": str(MODEL_PATH),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
):
    try:
        features = ChurnFeatures(
            gender=gender,
            SeniorCitizen=SeniorCitizen,
            Partner=Partner,
            Dependents=Dependents,
            tenure=tenure,
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges,
        )
        prediction, probability = _predict_score(features)
        result = {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk_level": _risk_label(probability),
        }
    except Exception as exc:
        if isinstance(exc, HTTPException):
            result = {
                "error": (
                    "Le modèle n'est pas chargé. Vérifie la présence du fichier "
                    "churn_xgb_pipeline.joblib puis redémarre l'application."
                )
            }
        else:
            result = {"error": "Erreur pendant le scoring. Vérifie les données saisies."}

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "model_loaded": model is not None,
            "model_path": str(MODEL_PATH),
        },
    )
