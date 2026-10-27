"""Predict the Iris species for one flower measurement."""
import sys

import joblib
from sklearn.datasets import load_iris

from src.train import MODEL_PATH


def predict(features):
    model = joblib.load(MODEL_PATH)
    return load_iris().target_names[model.predict([features])[0]]


if __name__ == "__main__":
    values = [float(v) for v in sys.argv[1:5]] or [5.1, 3.5, 1.4, 0.2]
    print(f"Prediction for {values}: {predict(values)}")
