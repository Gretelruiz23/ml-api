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
    """Busca recursivamente un objeto del tipo cls dentro de Pipelines/ColumnTransformers."""
    if isinstance(obj, cls):
        return obj
    steps = getattr(obj, "steps", None)
    if steps:
        for _, step in steps:
            found = _find_estimator(step, cls)
            if found is not None:
                return found
    if isinstance(obj, ColumnTransformer) or obj.__class__.__name__ == "ColumnTransformer":
        for _, trans, _ in getattr(obj, "transformers_", []):
            found = _find_estimator(trans, cls)
            if found is not None:
                return found
    return None

def _find_preprocessor(obj):
    return _find_estimator(obj, ColumnTransformer)

def _get_transformed_feature_names(preproc: ColumnTransformer):
    """
    Devuelve:
      - names_out: nombres de columnas transformadas (strings)
      - group_of : por cada col transformada, el nombre de la columna ORIGINAL
                   (sirve para sumar contribuciones de one-hot al feature base)
    Soporta casos donde el CT fue entrenado con índices (sin nombres).
    """
    def to_in_features(cols):
        if isinstance(cols, slice):
            idxs = list(range(cols.start or 0, cols.stop, cols.step or 1))
        elif isinstance(cols, (list, tuple, np.ndarray)):
            idxs = list(cols)
        else:
            idxs = [cols]
        feats = []
        for c in idxs:
            if isinstance(c, (int, np.integer)):
                if FEATURE_ORDER and 0 <= int(c) < len(FEATURE_ORDER):
                    feats.append(str(FEATURE_ORDER[int(c)]))
                else:
                    feats.append(str(c))
            else:
                feats.append(str(c))
        return feats

    names_out, group_of = [], []

    for name, trans, cols in preproc.transformers_:
        if name == "remainder":
            continue

        fns = None
        try:
            in_features = to_in_features(cols)
            # muchos transformadores modernos soportan esto; Pipeline puede fallar
            fns = list(trans.get_feature_names_out(in_features))
        except Exception:
            fns = None

        if fns is None:
            in_features = to_in_features(cols)
            # Fallback SIN expansión (imputer/escalers). OJO: puede ser menor que X.shape[1]
            fns = in_features[:]

        for fn in fns:
            s = str(fn)
            base = None
            for col in sorted(cat_cols, key=len, reverse=True):
                if s.startswith(col + "_"):
                    base = col
                    break
            if base is None:
                base = s
            names_out.append(s)
            group_of.append(base)

    return names_out, group_of

def _split_pre_and_rf(model):
    """
    Devuelve (pre, rf) donde:
      - pre: pipeline/transformador con SOLO etapas que tengan .transform (se saltan resamplers tipo SMOTE/SMOTENC)
      - rf : RandomForestClassifier final
    No depende de la clase exacta del pipeline (sklearn/imblearn).
    """
    steps = getattr(model, "steps", None)
    if not steps:
        raise HTTPException(status_code=400, detail="El modelo cargado no es un Pipeline o no expone 'steps'.")

    rf_idx = None
    for i, (name, step) in enumerate(steps):
        if isinstance(step, RandomForestClassifier):
            rf_idx = i
            break
    if rf_idx is None:
        raise HTTPException(status_code=400, detail="No se encontró RandomForestClassifier dentro del Pipeline.")
    if rf_idx == 0:
        raise HTTPException(status_code=400, detail="No hay etapas de preprocesamiento antes del RandomForest.")

    pre_steps = []
    for name, step in steps[:rf_idx]:
        if hasattr(step, "transform"):
            pre_steps.append((name, step))
        else:
            # resamplers (SMOTE/SMOTENC/etc.) carecen de .transform; se saltan en inferencia
            pass

    if not pre_steps:
        ct = _find_preprocessor(model)
        if ct is None:
            raise HTTPException(status_code=400, detail="No hay preprocesador con 'transform' antes del RF.")
        pre = ct
    else:
        pre = Pipeline(steps=pre_steps)

    rf = steps[rf_idx][1]
    return pre, rf

# ===== Esquemas =====
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de observaciones {feature: valor}")
    return_proba: bool = Field(False, description="Si True, devuelve probabilidades (si el modelo soporta)")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[int]
    probabilities: Optional[List[Dict[str, float]]] = None
    model_info: Dict[str, Any]

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

    # 1) orden/tipos como en /predict
    df, _ = ensure_feature_order(payload.records)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[cat_cols] = df[cat_cols].astype(str).fillna("")

    # 2) PARTIR EL PIPELINE REAL: pre + rf
    pre, rf = _split_pre_and_rf(MODEL)

    # 3) transformar con EL MISMO preprocesamiento que usó el modelo al entrenarse
    X = pre.transform(df)  # puede ser sparse
    X_ti = X.toarray() if hasattr(X, "toarray") else np.asarray(X)  # densificar para treeinterpreter/RF

    # 4) intentar recuperar el ColumnTransformer dentro de 'pre' (para mapear a columnas base)
    ct = _find_preprocessor(pre)
    if ct is not None:
        _, group_of = _get_transformed_feature_names(ct)
    else:
        n_out = X_ti.shape[1]
        group_of = [str(FEATURE_ORDER[i]) if FEATURE_ORDER and i < len(FEATURE_ORDER) else f"f{i}" for i in range(n_out)]

    # Ajustar longitud por seguridad (evita IndexError si faltan nombres)
    if len(group_of) < X_ti.shape[1]:
        group_of += [f"f{j}" for j in range(len(group_of), X_ti.shape[1])]
    elif len(group_of) > X_ti.shape[1]:
        group_of = group_of[:X_ti.shape[1]]

    # 5) predicción y contribuciones locales
    proba = rf.predict_proba(X_ti)
    _, bias, contribs = ti.predict(rf, X_ti)
    classes = getattr(rf, "classes_", np.arange(proba.shape[1]))

    items: List[ExplainItem] = []
    for i in range(X_ti.shape[0]):
        cls_idx = int(np.argmax(proba[i]))
        contrib_vec = contribs[i, :, cls_idx]  # n_features_transformed

        # sumamos contribuciones de one-hot a la columna base
        agg: Dict[str, float] = {}
        for j, g in enumerate(group_of):
            agg[g] = agg.get(g, 0.0) + float(contrib_vec[j])

        # valores originales (antes de preprocesar), para mostrar junto al driver
        val_sample: Dict[str, Any] = {col: df.iloc[i][col] for col in df.columns}

        top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        top_list: List[TopFeature] = [
            TopFeature(feature=k, contrib=float(v), valor=_to_py(val_sample.get(k, None)) if k in df.columns else None)
            for k, v in top
        ]

        items.append(ExplainItem(
            prediccion=int(classes[cls_idx]) if hasattr(classes[cls_idx], "__int__") else int(cls_idx),
            probabilidades={str(int(classes[j])): float(proba[i, j]) for j in range(len(classes))},
            top_features=top_list,
            rules=None
        ))

    return ExplainResponse(
        items=items,
        model_info=get_model_info()
    )
