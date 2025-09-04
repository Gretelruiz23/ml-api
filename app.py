# app.py
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib, json, os
import pandas as pd
import requests, pathlib, shutil

# ---------- Config ----------
MODEL_PATH    = os.getenv("MODEL_PATH", "modelo/modelo_rf_final.pkl")
MODEL_URL     = os.getenv("MODEL_URL")  
METADATA_PATH = os.getenv("METADATA_PATH", "modelo/metadata.json")  
API_TOKEN     = os.getenv("API_TOKEN", None)  # opcional

app = FastAPI(
    title="RF Prediction API",
    version="1.0.0",
    description="API para servir un modelo de Random Forest (FastAPI)",
)

MODEL = None
FEATURE_ORDER: Optional[List[str]] = None
CLASSES: Optional[List[Any]] = None

# ===== Esquemas =====
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de observaciones {feature: valor}")
    return_proba: bool = Field(False, description="Si True, devuelve probabilidades (si el modelo soporta)")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    model_info: Dict[str, Any]

# ===== Descarga segura del modelo si no existe =====
def ensure_model_present():
    path = pathlib.Path(MODEL_PATH)
    if path.exists() and path.stat().st_size > 0:
        print(f"[startup] Modelo ya existe: {path} ({path.stat().st_size/1_048_576:.1f} MB)")
        return
    if not MODEL_URL:
        raise RuntimeError(f"[startup] Modelo no encontrado en {MODEL_PATH} y no hay MODEL_URL configurada.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"[startup] Descargando modelo desde: {MODEL_URL}")
    headers = {"User-Agent": "railway-fastapi/1.0", "Accept": "*/*"}
    with requests.get(MODEL_URL, stream=True, timeout=600, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    size_mb = tmp.stat().st_size / 1_048_576
    if size_mb < 5:
        raise RuntimeError(f"[startup] Tamaño anómalo tras descargar: {size_mb:.2f} MB (¿copiaste el link directo del asset?)")
    tmp.replace(path)
    print(f"[startup] Descarga OK -> {path} ({size_mb:.1f} MB)")

# ===== Startup =====
@app.on_event("startup")
def load_model_and_meta():
    # Diagnóstico opcional: saltar carga
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        print("[startup] SKIP_MODEL_LOAD=1 -> no se carga el modelo (diagnóstico).")
        return

    # 1) Asegura modelo presente (descarga si falta)
    ensure_model_present()

    # 2) Carga modelo
    global MODEL, FEATURE_ORDER, CLASSES
    try:
        print("[startup] Cargando modelo con joblib...")
        MODEL = joblib.load(MODEL_PATH)
        print(f"[startup] Modelo cargado: {type(MODEL).__name__}")
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}")

    # 3) Metadata (opcional)
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            FEATURE_ORDER = meta.get("feature_order") or FEATURE_ORDER
            CLASSES = meta.get("classes") or CLASSES
            print("[startup] Metadata cargada.")
        except Exception as e:
            print(f"[startup] Advertencia: no se pudo leer metadata: {e}")

    # 4) Fallback names/clases desde el modelo
    if FEATURE_ORDER is None and hasattr(MODEL, "feature_names_in_"):
        FEATURE_ORDER = list(MODEL.feature_names_in_)
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
    order = FEATURE_ORDER or list(df.columns)
    faltantes = [f for f in order if f not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan features requeridos: {faltantes}. "
                   f"Se esperan exactamente {len(order)} features: {order}"
        )
    df = df.reindex(columns=order)
    return df, order

def get_model_info():
    return {
        "model_type": (type(MODEL).__name__ if MODEL is not None else None),
        "supports_proba": (hasattr(MODEL, "predict_proba") if MODEL is not None else None),
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
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    df, _ = ensure_feature_order(payload.records)

    try:
        preds = MODEL.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predict: {e}")

    probas = None
    if payload.return_proba:
        if hasattr(MODEL, "predict_proba"):
            try:
                raw = MODEL.predict_proba(df)
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
