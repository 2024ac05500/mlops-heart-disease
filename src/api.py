from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.utils import load_model

app = FastAPI()

class Input(BaseModel):
    features: list

MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model("models/model.joblib")
    except Exception:
        MODEL = None

@app.post("/predict")
def predict_endpoint(inp: Input):
    if MODEL is None:
        return {"error": "model not available"}
    arr = np.array(inp.features).reshape(1, -1)
    prob = MODEL.predict_proba(arr)[0, 1]
    return {"probability": float(prob)}
