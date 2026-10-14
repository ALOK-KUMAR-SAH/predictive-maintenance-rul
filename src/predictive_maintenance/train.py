import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import DEFAULT_TARGET_CAP, MODEL_DIR, REPORT_DIR, RANDOM_STATE, ensure_project_dirs
from .data import add_rul_target, load_cmapss, make_demo_data, split_by_engine
from .features import select_features
from .models import make_pipeline, model_factories, save_artifact


def train_and_evaluate(frame: pd.DataFrame, target_cap: int = DEFAULT_TARGET_CAP) -> dict:
    labelled = add_rul_target(frame, cap=target_cap)
    train, test = split_by_engine(labelled, random_state=RANDOM_STATE)
    x_train, x_test = select_features(train), select_features(test)
    y_train, y_test = train["rul"], test["rul"]
    metrics = {}
    fitted = {}
    for name, estimator in model_factories(RANDOM_STATE).items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        }
        fitted[name] = pipeline
    best_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    metadata = {
        "best_model": best_name,
        "features": list(x_train.columns),
        "target": "rul",
        "target_cap": target_cap,
        "train_engines": sorted(train["unit_id"].unique().tolist()),
        "test_engines": sorted(test["unit_id"].unique().tolist()),
        "metrics": metrics,
    }
    save_artifact(MODEL_DIR / "best_model.joblib", fitted[best_name], metadata)
    joblib.dump(
        fitted[best_name].named_steps["preprocessor"],
        MODEL_DIR / "preprocessor.joblib",
    )
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe C-MAPSS RUL regressors.")
    parser.add_argument("--data", default="data/raw/train_FD001.txt")
    parser.add_argument("--demo", action="store_true", help="Use deterministic synthetic smoke-test data.")
    parser.add_argument("--target-cap", type=int, default=DEFAULT_TARGET_CAP)
    args = parser.parse_args()
    ensure_project_dirs()
    frame = make_demo_data() if args.demo else load_cmapss(args.data)
    metadata = train_and_evaluate(frame, args.target_cap)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
