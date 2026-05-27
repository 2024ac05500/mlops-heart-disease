import pandas as pd
from src.data_preprocessing import preprocess


def test_preprocess_removes_nans():
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    out = preprocess(df)
    assert out.isnull().sum().sum() == 0
    assert "a" in out.columns and "b" in out.columns
