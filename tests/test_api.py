import sys
from pathlib import Path

# ensure repo root is on path so `src` can be importable
repo_root = next((p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient  # noqa: E402
from src.api import app  # noqa: E402

# sample input features in expected RAW_FEATURE_COLUMNS order
sample = {
    "features": [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]
}


def test_predict_endpoint():
    with TestClient(app) as client:
        resp = client.post("/predict", json=sample)
        assert resp.status_code == 200
        json_data = resp.json()
        assert "prediction" in json_data
        assert "confidence" in json_data
        assert isinstance(json_data["prediction"], int)
        assert json_data["confidence"] is None or isinstance(json_data["confidence"], float)
