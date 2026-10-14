import pandas as pd
import pytest

from predictive_maintenance.data import add_rul_target, make_demo_data, split_by_engine
from predictive_maintenance.features import select_features
from predictive_maintenance.models import make_pipeline, model_factories
from predictive_maintenance.predict import predict_rul


def test_rul_is_zero_at_last_engine_cycle():
    frame = add_rul_target(make_demo_data(n_engines=2), cap=None)
    last = frame.loc[frame.groupby("unit_id")["cycle"].idxmax()]
    assert last["rul"].tolist() == [0, 0]


def test_split_keeps_engines_disjoint():
    train, test = split_by_engine(make_demo_data(n_engines=8), test_size=0.25)
    assert set(train.unit_id).isdisjoint(set(test.unit_id))


def test_preprocessing_and_prediction():
    frame = add_rul_target(make_demo_data(n_engines=6))
    train, test = split_by_engine(frame)
    pipeline = make_pipeline(model_factories()["linear_regression"])
    pipeline.fit(select_features(train), train.rul)
    predictions = predict_rul({"pipeline": pipeline}, select_features(test).head(3))
    assert len(predictions) == 3
    assert all(value >= 0 for value in predictions)


def test_invalid_cap_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        add_rul_target(make_demo_data(n_engines=2), cap=0)


def test_missing_features_are_rejected():
    with pytest.raises(ValueError, match="Missing feature"):
        select_features(pd.DataFrame({"cycle": [1]}))
