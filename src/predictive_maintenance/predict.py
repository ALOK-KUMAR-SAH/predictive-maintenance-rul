import pandas as pd

from .features import select_features


def predict_rul(artifact: dict, records: pd.DataFrame) -> list[float]:
    """Predict RUL for one or more records using a loaded artifact."""
    if "pipeline" not in artifact:
        raise ValueError("Artifact does not contain a pipeline.")
    values = artifact["pipeline"].predict(select_features(records))
    return [max(0.0, float(value)) for value in values]
