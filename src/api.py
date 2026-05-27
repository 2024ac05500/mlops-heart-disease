from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.utils import load_model
from pathlib import Path
import glob

app = FastAPI()


class Input(BaseModel):
    features: list


MODEL = None


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


@app.on_event("startup")
def startup_event():
    global MODEL
    model, path = _try_load_candidates()
    MODEL = model
    if MODEL is not None:
        print(f"Loaded model from {path}")
    else:
        print("No model found in models/; predict endpoint will return error until a model is placed.")


@app.post("/predict")
def predict_endpoint(inp: Input):
    if MODEL is None:
        return {"error": "model not available"}
    arr = np.array(inp.features).reshape(1, -1)
    # try predict_proba, fall back to decision_function
    try:
        prob = MODEL.predict_proba(arr)[0, 1]
    except Exception:
        try:
            p = MODEL.decision_function(arr)
            prob = float(p.ravel()[0])
        except Exception:
            prob = None

    result = {}
    if prob is None:
        result["error"] = "model does not provide probability or decision function"
    else:
        result["probability"] = float(prob)
        result["prediction"] = int(prob >= 0.5)
    return result
