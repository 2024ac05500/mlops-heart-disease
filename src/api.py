# Purpose: Serve FastAPI prediction and monitoring endpoints.
import logging
import time
from pathlib import Path
import glob

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from src.utils import load_model
from joblib import load as joblib_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
logger = logging.getLogger("heart-disease-api")

REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Total HTTP requests processed by the heart disease API",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_latency_seconds",
    "Request latency in seconds for heart disease API",
    ["method", "path"],
)
ERROR_COUNT = Counter(
    "heart_disease_api_errors_total",
    "Total number of HTTP error responses from the heart disease API",
    ["method", "path", "status_code"],
)

app = FastAPI()


class Input(BaseModel):
    features: list


MODEL = None
PREPROCESSOR = None
# expected raw input columns (order expected by the dataset)
RAW_FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def _try_load_candidates():
    """Try multiple candidate model paths and return the first successfully loaded model."""
    candidates = [
        Path("models") / "model.joblib",
        Path("models") / "best_model.joblib",
    ]
    # add any model_*.joblib files
    for p in sorted(glob.glob(str(Path("models") / "model_*.joblib"))):
        candidates.append(Path(p))

    for c in candidates:
        try:
            if c.exists():
                m = load_model(str(c))
                return m, str(c)
        except Exception:
            continue
    return None, None


def _try_load_preprocessor():
    ppath = Path("models") / "preprocessor.joblib"
    if ppath.exists():
        try:
            p = joblib_load(str(ppath))
            return p, str(ppath)
        except Exception:
            return None, None
    return None, None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = Response(status_code=500)
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.time() - start_time
        method = request.method
        path = request.url.path
        status_code = str(response.status_code)

        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
        if response.status_code >= 400:
            ERROR_COUNT.labels(method=method, path=path, status_code=status_code).inc()

        logger.info(
            "%s %s %s %.4fs",
            method,
            path,
            status_code,
            elapsed,
        )


@app.on_event("startup")
def startup_event():
    global MODEL
    model, path = _try_load_candidates()
    preproc, ppath = _try_load_preprocessor()
    MODEL = model
    global PREPROCESSOR
    PREPROCESSOR = preproc
    if MODEL is not None:
        logger.info("Loaded model from %s", path)
    else:
        logger.warning(
            "No model found in models/; predict endpoint will return error until a model is placed."
        )
    if PREPROCESSOR is not None:
        logger.info("Loaded preprocessor from %s", ppath)
    else:
        logger.warning(
            "No preprocessor found in models/; incoming raw features must be pre-transformed to match model input."
        )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict_endpoint(inp: Input):
    if MODEL is None:
        return {"error": "model not available"}
    # build DataFrame from raw features with expected column order
    try:
        import pandas as pd

        df = pd.DataFrame([inp.features], columns=RAW_FEATURE_COLUMNS)
    except Exception:
        # fallback to numpy if DataFrame construction fails
        df = None

    if PREPROCESSOR is not None and df is not None:
        X = PREPROCESSOR.transform(df)
    else:
        # assume caller provided already-transformed features
        X = np.array(inp.features).reshape(1, -1)
    # try predict_proba, fall back to decision_function; otherwise return prediction with null confidence
    prob = None
    try:
        proba = MODEL.predict_proba(X)
        # if binary classifier, take column for positive class
        if proba.shape[1] == 2:
            prob = float(proba[0, 1])
        else:
            # multiclass: take max class probability
            prob = float(proba[0].max())
    except Exception:
        try:
            p = MODEL.decision_function(X)
            prob = float(p.ravel()[0])
        except Exception:
            prob = None

    result = {}
    if prob is None:
        # fallback: return predicted label and unknown confidence
        # try numpy array first, then try pandas DataFrame (some pipelines expect DataFrame)
        try:
            pred = MODEL.predict(X)[0]
            result["prediction"] = int(pred)
            result["confidence"] = None
        except Exception:
            try:
                import pandas as pd

                if df is not None:
                    pred = MODEL.predict(df)[0]
                    result["prediction"] = int(pred)
                    result["confidence"] = None
                else:
                    result["error"] = "model does not provide prediction/probability interface"
            except Exception:
                result["error"] = "model does not provide prediction/probability interface"
    else:
        result["prediction"] = int(prob >= 0.5)
        result["confidence"] = float(prob)
    return result
