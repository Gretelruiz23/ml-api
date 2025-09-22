# app.py
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import os, json, pathlib, shutil, requests, pickle, joblib
import pandas as pd
import numpy as np

# ---------- Config ----------
MODEL_PATH     = os.getenv("MODEL_PATH", "modelo/modelo_RandomForest_VAL_final.pkl")
UMBRAL_PATH    = os.getenv("UMBRAL_PATH", "modelo/umbrales_RandomForest_VAL_final.pkl")
METADATA_PATH  = os.getenv("METADATA_PATH", "modelo/metadata_modelo_valoracion.json")  
MODEL_URL      = os.getenv("MODEL_URL")  
API_TOKEN      = os.getenv("API_TOKEN", None)  # opcional

app = FastAPI(
    title="Modelo de Predicción API",
    version="1.1.0",
    description="API para servir el modelo ganador (DT/RF/XGB) con umbral ajustable",
)

MODEL = None
FEATURE_ORDER: Optional[List[str]] = None
CLASSES: Optional[List[Any]] = None
THRESHOLDS: Optional[Dict[str, float]] = None


# ======== columnas ======= 

usable_cols = [
    "periodo_1","periodo_2","periodo_3","prom_p123","std_p123","trend_p3_p1",
    "lag1_periodo_1","lag1_periodo_2","lag1_periodo_3",
    "lag1_prom_p123","lag1_std_p123","lag1_trend_p3_p1","lag1_nota_final",
    "PUNTAJE_SABER11",
    "SEDE","ZONA","JORNADA","GRUPO","TAMANO_GRUPO","GRADO","ANO","GENERO",
    "ASIGNATURA","INTERNET_COLEGIO","BIBLIOTECA","LABORATORIO_CIENCIAS_INFORMATICA",
    "CLASIFICACION_SABER11","PAE"
]
num_cols = [
    "periodo_1","periodo_2","periodo_3","prom_p123","std_p123","trend_p3_p1",
    "lag1_periodo_1","lag1_periodo_2","lag1_periodo_3",
    "lag1_prom_p123","lag1_std_p123","lag1_trend_p3_p1","lag1_nota_final",
    "PUNTAJE_SABER11"
]
cat_cols = [
    "SEDE","ZONA","JORNADA","GRUPO","TAMANO_GRUPO","GRADO","ANO","GENERO",
    "ASIGNATURA","INTERNET_COLEGIO","BIBLIOTECA","LABORATORIO_CIENCIAS_INFORMATICA",
    "CLASIFICACION_SABER11","PAE"
]

def prepare_X_from_records(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # 1) Validación: que vengan TODAS las columnas que el modelo espera
    missing = [c for c in usable_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas: {missing}. Se esperan {len(usable_cols)} columnas: {usable_cols}"
        )

    # 2) Tomar SOLO las columnas esperadas y en el ORDEN correcto
    X = df[usable_cols].copy()

    # 3) Tipos correctos para el pipeline
    #    - num_cols → numérico (deja NaN si no se puede; el imputer del pipeline se encarga)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    #    - cat_cols → string (sin NaN)
    X[cat_cols] = X[cat_cols].fillna("").astype(str)

    return X

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
    headers = {"User-Agent": "fastapi-model/1.0", "Accept": "*/*"}
    with requests.get(MODEL_URL, stream=True, timeout=600, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    size_mb = tmp.stat().st_size / 1_048_576
    if size_mb < 0.1:
        raise RuntimeError(f"[startup] Tamaño anómalo tras descargar: {size_mb:.2f} MB")
    tmp.replace(path)
    print(f"[startup] Descarga OK -> {path} ({size_mb:.1f} MB)")

# ===== Startup =====
@app.on_event("startup")
def load_model_and_meta():
    global MODEL, FEATURE_ORDER, CLASSES, THRESHOLDS

    # Diagnóstico opcional: saltar carga
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        print("[startup] SKIP_MODEL_LOAD=1 -> no se carga el modelo (diagnóstico).")
        return

    # 1) Asegura modelo presente
    ensure_model_present()

    # 2) Carga modelo
    try:
        print("[startup] Intentando cargar modelo con joblib...")
        MODEL = joblib.load(MODEL_PATH)
        print(f"[startup] Modelo cargado con joblib: {type(MODEL).__name__}")
    except Exception as e1:
        try:
            print("[startup] Fallback: intentando cargar modelo con pickle...")
            with open(MODEL_PATH, "rb") as f:
                MODEL = pickle.load(f)
            print(f"[startup] Modelo cargado con pickle: {type(MODEL).__name__}")
        except Exception as e2:
            raise RuntimeError(f"No se pudo cargar el modelo ({MODEL_PATH}). joblib error={e1}, pickle error={e2}")

    # 3) Metadata
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

    # 5) Carga umbrales
    if os.path.exists(UMBRAL_PATH):
        try:
            with open(UMBRAL_PATH, "rb") as f:
                THRESHOLDS = pickle.load(f)
            print(f"[startup] Umbrales cargados: {THRESHOLDS}")
        except Exception as e:
            print(f"[startup] Advertencia: no se pudo cargar umbrales: {e}")

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

def apply_thresholds(probas: np.ndarray):
    """Aplica umbrales personalizados si están definidos."""
    if THRESHOLDS and CLASSES:
        preds = []
        for row in probas:
            row_dict = {str(c): float(p) for c, p in zip(CLASSES, row)}
            chosen = None
            for c, thr in THRESHOLDS.items():
                if row_dict.get(str(c), 0) >= thr:
                    chosen = c
                    break
            if chosen is None:
                chosen = CLASSES[np.argmax(row)]  # fallback
            preds.append(chosen)
        return preds
    else:
        return CLASSES[np.argmax(probas, axis=1)] if CLASSES else np.argmax(probas, axis=1)

def get_model_info():
    return {
        "model_type": (type(MODEL).__name__ if MODEL is not None else None),
        "supports_proba": (hasattr(MODEL, "predict_proba") if MODEL is not None else None),
        "n_expected_features": (len(FEATURE_ORDER) if FEATURE_ORDER else None),
        "feature_order": FEATURE_ORDER,
        "classes": CLASSES,
        "thresholds": THRESHOLDS,
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

    # Forzar tipos correctos para el ColumnTransformer
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")  # numéricas
    df[cat_cols] = df[cat_cols].astype(str).fillna("")                # categóricas como string


    probas = None
    preds = None

    if payload.return_proba:
        if hasattr(MODEL, "predict_proba"):
            try:
                raw = MODEL.predict_proba(df)
                probas = [{str(c): float(p) for c, p in zip(CLASSES, row)} for row in raw] if CLASSES else raw.tolist()
                preds = apply_thresholds(raw)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error en predict_proba: {e}")
        else:
            raise HTTPException(status_code=400, detail="El modelo no soporta probabilidades.")
    else:
        try:
            preds = MODEL.predict(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en predict: {e}")

    return PredictionResponse(
        predictions=[(p.tolist() if hasattr(p, "tolist") else p) for p in preds],
        probabilities=probas,
        model_info=get_model_info(),
    )
