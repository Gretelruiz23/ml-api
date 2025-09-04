   
# app.py
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib, json, os
import pandas as pd
import requests, pathlib, shutil

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "./modelo/modelo_rf_final.pkl")
MODEL_URL = os.getenv("sha256:e19420a7f1733b2cf3a237dfd66058eed5c8fe4d03bbb009dd2c250bbf1cb92a")
METADATA_PATH = os.getenv("METADATA_PATH", "./modelo/metadata.json")

# Token opcional simple (Bearer). Si no lo configuras, no valida.
API_TOKEN = os.getenv("API_TOKEN", None)

app = FastAPI(
    title="RF Prediction API",
    version="1.0.0",
    description="API para servir un modelo de Random Forest (FastAPI)",
)

# Variables globales
MODEL = None
FEATURE_ORDER: Optional[List[str]] = None
CLASSES: Optional[List[Any]] = None

# ===== Esquemas =====
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de observaciones {feature: valor}")
    return_proba: bool = Field(False, description="Si True, devuelve probabilidades (si el modelo soporta)")

class PredictionResponse(BaseModel):
    # Evita el warning de 'model_' namespace
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    model_info: Dict[str, Any]

# ===== Startup: carga modelo y metadata =====

def ensure_model_present():
    path = pathlib.Path(MODEL_PATH)
    if path.exists():
        return
    if not MODEL_URL:
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH} y no hay MODEL_URL configurada.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # descarga en streaming
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            shutil.copyfileobj(r.raw, f)


@app.on_event("startup")
def load_model_and_meta():
    global MODEL, FEATURE_ORDER, CLASSES
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}")

    # 1) Lee metadata.json si existe (recomendado)
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            FEATURE_ORDER = meta.get("feature_order") or FEATURE_ORDER
            CLASSES = meta.get("classes") or CLASSES
        except Exception:
            pass

    # 2) Si no hay metadata, intenta usar nombres del modelo
    if FEATURE_ORDER is None and hasattr(MODEL, "feature_names_in_"):
        FEATURE_ORDER = list(MODEL.feature_names_in_)

    # 3) Clases del modelo (si no vinieron en metadata)
    if CLASSES is None and hasattr(MODEL, "classes_"):
        try:
            CLASSES = list(MODEL.classes_)
        except Exception:
            CLASSES = None

# ===== Utilidades =====

def ensure_feature_order(records: List[Dict[str, Any]]):
    if not records:
        raise HTTPException(status_code=400, detail="No se recibieron registros (records).")

    df = pd.DataFrame(records)

    # Definir el orden esperado
    order = FEATURE_ORDER or list(df.columns)

    # Validar faltantes
    faltantes = [f for f in order if f not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan features requeridos: {faltantes}. "
                   f"Se esperan exactamente {len(order)} features: {order}"
        )

    # Reindexa en el orden correcto; ignora columnas extra
    df = df.reindex(columns=order)

    return df, order


def get_model_info():
    return {
        "model_type": type(MODEL).__name__,
        "supports_proba": hasattr(MODEL, "predict_proba"),
        "n_expected_features": (len(FEATURE_ORDER) if FEATURE_ORDER else None),
        "feature_order": FEATURE_ORDER,
        "classes": CLASSES,
    }

# ===== Endpoints =====
@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionRequest):
    df, _ = ensure_feature_order(payload.records)
    preds = MODEL.predict(df)


    try:
        preds = MODEL.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predict: {e}")

    probas = None
    if payload.return_proba:
        if hasattr(MODEL, "predict_proba"):
            try:
                raw = MODEL.predict_proba(df)
                # Mapea probabilidades a nombres de clase si los tenemos
                if CLASSES and len(CLASSES) == raw.shape[1]:
                    probas = [{str(c): float(p) for c, p in zip(CLASSES, row)} for row in raw]
                else:
                    probas = [{str(i): float(p) for i, p in enumerate(row)} for row in raw]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error en predict_proba: {e}")
        else:
            raise HTTPException(status_code=400, detail="El modelo no soporta probabilidades.")

    return PredictionResponse(
        predictions=[(p.tolist() if hasattr(p, "tolist") else p) for p in preds],
        probabilities=probas,
        model_info=get_model_info(),
    )