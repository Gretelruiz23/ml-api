# app.py
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import os, json, pathlib, shutil, requests, pickle, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from typing import List

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

FEATURE_ORDER = usable_cols

def prepare_X_from_records(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    missing = [c for c in usable_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas: {missing}. Se esperan {len(usable_cols)} columnas: {usable_cols}"
        )
    X = df[usable_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[cat_cols] = X[cat_cols].fillna("").astype(str)
    return X

# --- EXPLAIN: imports y helpers ---
try:
    from treeinterpreter import treeinterpreter as ti
    HAS_TI = True
except Exception:
    HAS_TI = False

def _find_estimator(obj, cls):
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            found = _find_estimator(step, cls)
            if found is not None:
                return found
    if isinstance(obj, ColumnTransformer):
        for _, trans, _ in getattr(obj, "transformers_", []):
            found = _find_estimator(trans, cls)
            if found is not None:
                return found
    return None

def _find_preprocessor(obj):
    return _find_estimator(obj, ColumnTransformer)

def _get_transformed_feature_names(preproc: ColumnTransformer):
    names_out, group_of = [], []
    for name, trans, cols in preproc.transformers_:
        if name == "remainder":
            continue
        try:
            fns = list(trans.get_feature_names_out(cols))
        except Exception:
            fns = list(cols)
        for fn in fns:
            base = fn.rsplit("_", 1)[0] if "_" in fn else fn
            names_out.append(str(fn))
            group_of.append(str(base))
    return names_out, group_of

# ===== Esquemas =====
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de observaciones {feature: valor}")
    return_proba: bool = Field(False, description="Si True, devuelve probabilidades (si el modelo soporta)")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[int]
    probabilities: Optional[List[Dict[str, float]]] = None
    model_info: Dict[str, Any]

# --- EXPLAIN: esquemas de respuesta ---
class TopFeature(BaseModel):
    feature: str
    contrib: float
    valor: Optional[Any] = None

class ExplainItem(BaseModel):
    prediccion: int
    probabilidades: Optional[Dict[str, float]] = None
    top_features: List[TopFeature]
    rules: Optional[List[str]] = None  # opcional

class ExplainResponse(BaseModel):
    items: List[ExplainItem]
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

    if os.getenv("SKIP_MODEL_LOAD") == "1":
        print("[startup] SKIP_MODEL_LOAD=1 -> no se carga el modelo (diagnóstico).")
        return

    ensure_model_present()

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

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            FEATURE_ORDER = meta.get("feature_order") or FEATURE_ORDER
            CLASSES = meta.get("classes") or CLASSES
            print("[startup] Metadata cargada.")
        except Exception as e:
            print(f"[startup] Advertencia: no se pudo leer metadata: {e}")

    if FEATURE_ORDER is None and hasattr(MODEL, "feature_names_in_"):
        FEATURE_ORDER = list(MODEL.feature_names_in_)
    if CLASSES is None and hasattr(MODEL, "classes_"):
        try:
            CLASSES = list(MODEL.classes_)
        except Exception:
            CLASSES = None

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
    if THRESHOLDS and CLASSES is not None:
        preds = []
        for row in probas:
            row_dict = {str(c): float(p) for c, p in zip(CLASSES, row)}
            chosen = None
            for c, thr in THRESHOLDS.items():
                if row_dict.get(str(c), 0) >= thr:
                    chosen = c
                    break
            if chosen is None:
                chosen = CLASSES[np.argmax(row)]
            preds.append(chosen)
        return preds
    else:
        return CLASSES[np.argmax(probas, axis=1)] if CLASSES else np.argmax(probas, axis=1)

def _to_py(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    return obj

def get_model_info():
    info = {
        "model_type": (type(MODEL).__name__ if MODEL is not None else None),
        "supports_proba": (hasattr(MODEL, "predict_proba") if MODEL is not None else None),
        "n_expected_features": (len(FEATURE_ORDER) if FEATURE_ORDER else None),
        "feature_order": FEATURE_ORDER,
        "classes": CLASSES,
        "thresholds": THRESHOLDS,
    }
    return _to_py(info)

# ===== Endpoints =====
@app.get("/health", tags=["infra"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    df, _ = ensure_feature_order(payload.records)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[cat_cols] = df[cat_cols].astype(str).fillna("")

    probas = None
    preds: List[int]

    if payload.return_proba:
        if hasattr(MODEL, "predict_proba"):
            try:
                raw = MODEL.predict_proba(df)  # ndarray [n_samples, n_classes]
                n_classes = raw.shape[1]
                preds = [int(np.argmax(row)) for row in raw]
                probas = [{str(j): float(row[j]) for j in range(n_classes)} for row in raw]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error en predict_proba: {e}")
        else:
            raise HTTPException(status_code=400, detail="El modelo no soporta probabilidades.")
    else:
        try:
            preds = [int(p) for p in MODEL.predict(df)]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en predict: {e}")

    return PredictionResponse(
        predictions=preds,
        probabilities=probas,
        model_info=get_model_info(),
    )

# --- EXPLAIN: endpoint /explain ---
@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(payload: PredictionRequest, top_k: int = 6):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    if not HAS_TI:
        raise HTTPException(status_code=503, detail="treeinterpreter no instalado. Agrega 'treeinterpreter' a requirements.txt.")

    df, _ = ensure_feature_order(payload.records)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[cat_cols] = df[cat_cols].astype(str).fillna("")

    preproc = _find_preprocessor(MODEL)
    rf = _find_estimator(MODEL, RandomForestClassifier)
    if preproc is None or rf is None:
        raise HTTPException(status_code=400, detail="No se encontró ColumnTransformer y/o RandomForest en el modelo.")

    X = preproc.transform(df)
    _, group_of = _get_transformed_feature_names(preproc)

    proba = rf.predict_proba(X)
    _, bias, contribs = ti.predict(rf, X)
    classes = getattr(rf, "classes_", np.arange(proba.shape[1]))

    items: List[ExplainItem] = []
    for i in range(X.shape[0]):
        cls_idx = int(np.argmax(proba[i]))
        contrib_vec = contribs[i, :, cls_idx]

        agg: Dict[str, float] = {}
        val_sample: Dict[str, Any] = {}
        for j, g in enumerate(group_of):
            agg[g] = agg.get(g, 0.0) + float(contrib_vec[j])
        for col in df.columns:
            val_sample[col] = df.iloc[i][col]

        top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        top_list: List[TopFeature] = [
            TopFeature(feature=k, contrib=float(v), valor=_to_py(val_sample.get(k, None)) if k in df.columns else None)
            for k, v in top
        ]

        item = ExplainItem(
            prediccion=int(classes[cls_idx]) if hasattr(classes[cls_idx], "__int__") else int(cls_idx),
            probabilidades={str(int(classes[j])): float(proba[i, j]) for j in range(len(classes))},
            top_features=top_list,
            rules=None
        )
        items.append(item)

    return ExplainResponse(
        items=items,
        model_info=get_model_info()
    )
