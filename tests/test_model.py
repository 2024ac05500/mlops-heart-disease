from sklearn.datasets import make_classification
from src.train import train_model
from src.predict import predict


def test_train_and_predict(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    model_path = tmp_path / "model.joblib"
    model = train_model(X, y, out_path=str(model_path))
    probs = predict(str(model_path), X[:5])
    assert len(probs) == 5
    assert (probs >= 0).all() and (probs <= 1).all()
