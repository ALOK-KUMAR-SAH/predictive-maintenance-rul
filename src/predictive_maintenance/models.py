from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def model_factories(random_state: int = 42) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, random_state=random_state
        ),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])


def save_artifact(path: str | Path, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, destination)


def load_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact not found: {source}. Run the training command first.")
    artifact = joblib.load(source)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid artifact: expected a dictionary containing a fitted pipeline.")
    return artifact
