"""Train a Decision Tree on the Iris dataset and save it to model/."""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
