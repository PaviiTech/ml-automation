# train_model.py
import cProfile
import pstats
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    print("Starting training...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.3f}")

    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/iris_model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    return model_path

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    model_path = train_model()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats("artifacts/profile_stats.prof")
    print("Saved profiler data to artifacts/profile_stats.prof")
