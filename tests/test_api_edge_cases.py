# Purpose: Define automated tests for project functionality.
import numpy as np
import pandas as pd

import src.api as api_mod


def test_predict_endpoint_model_missing():
    api_mod.MODEL = None
    api_mod.PREPROCESSOR = None
    out = api_mod.predict_endpoint(api_mod.Input(features=[0] * 13))
    assert out["error"] == "model not available"


def test_predict_endpoint_uses_preprocessor_and_multiclass_proba():
    class _Preproc:
        def transform(self, df):
            assert isinstance(df, pd.DataFrame)
            return np.ones((1, 5))

    class _Model:
        def predict_proba(self, X):
            assert X.shape == (1, 5)
            return np.array([[0.1, 0.2, 0.7]])

    api_mod.PREPROCESSOR = _Preproc()
    api_mod.MODEL = _Model()

    out = api_mod.predict_endpoint(api_mod.Input(features=[1] * 13))
    assert out["prediction"] == 1
    assert out["confidence"] == 0.7


def test_predict_endpoint_decision_function_fallback():
    class _Model:
        def predict_proba(self, _X):
            raise RuntimeError("no proba")

        def decision_function(self, _X):
            return np.array([-0.2])

    api_mod.PREPROCESSOR = None
    api_mod.MODEL = _Model()

    out = api_mod.predict_endpoint(api_mod.Input(features=[1] * 13))
    assert out["prediction"] == 0
    assert out["confidence"] == -0.2


def test_predict_endpoint_predict_df_fallback_when_numpy_predict_fails():
    class _Model:
        def predict_proba(self, _X):
            raise RuntimeError("no proba")

        def decision_function(self, _X):
            raise RuntimeError("no decision")

        def predict(self, X):
            if isinstance(X, np.ndarray):
                raise RuntimeError("numpy path fails")
            if isinstance(X, pd.DataFrame):
                return np.array([1])
            raise RuntimeError("unexpected type")

    api_mod.PREPROCESSOR = None
    api_mod.MODEL = _Model()

    out = api_mod.predict_endpoint(api_mod.Input(features=[1] * 13))
    assert out["prediction"] == 1
    assert out["confidence"] is None


def test_predict_endpoint_returns_interface_error_when_all_methods_fail():
    class _Model:
        def predict_proba(self, _X):
            raise RuntimeError("no proba")

        def decision_function(self, _X):
            raise RuntimeError("no decision")

        def predict(self, _X):
            raise RuntimeError("no predict")

    api_mod.PREPROCESSOR = None
    api_mod.MODEL = _Model()

    out = api_mod.predict_endpoint(api_mod.Input(features=[1] * 13))
    assert out["error"] == "model does not provide prediction/probability interface"
