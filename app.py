import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predictive_maintenance.config import FEATURE_COLUMNS, MODEL_DIR, REPORT_DIR
from predictive_maintenance.data import RAW_COLUMNS, load_cmapss
from predictive_maintenance.models import load_artifact
from predictive_maintenance.predict import predict_rul

st.set_page_config(page_title="Predictive Maintenance | RUL", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance and RUL Prediction")
st.caption("NASA C-MAPSS FD001 benchmark workflow — estimates are not maintenance decisions.")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("View", ["Overview", "Dataset", "Evaluation", "Prediction"])

if section == "Overview":
    st.header("Project overview")
    st.markdown(
        "This application estimates **Remaining Useful Life (RUL)** in operating cycles using "
        "a leakage-aware regression pipeline. It compares linear regression, random forest, and "
        "histogram gradient boosting."
    )
    st.warning("The simulated C-MAPSS benchmark is not production telemetry. Validate on your fleet before taking action.")
    st.subheader("Regression versus classification")
    st.write("RUL regression predicts a continuous number of cycles. Failure classification predicts a discrete risk label within a chosen horizon.")
elif section == "Dataset":
    st.header("Dataset information")
    st.write("Expected source: NASA C-MAPSS FD001, extracted as data/raw/train_FD001.txt.")
    path = Path("data/raw/train_FD001.txt")
    if not path.exists():
        st.info("No local NASA file found. Run the download/extraction steps in README.md.")
    else:
        try:
            frame = load_cmapss(path)
            st.write(f"{len(frame):,} rows, {frame['unit_id'].nunique()} engines.")
            st.dataframe(frame.head(20), use_container_width=True)
            selected = st.selectbox("Sensor to plot", [c for c in RAW_COLUMNS if c.startswith("sensor_")])
            st.line_chart(frame[frame.unit_id <= 5].set_index("cycle")[selected])
        except (OSError, ValueError) as error:
            st.error(str(error))
elif section == "Evaluation":
    st.header("Model evaluation")
    metrics_path = REPORT_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation report found. Train a model first.")
    else:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.success(f"Best model by validation RMSE: {report['best_model']}")
        table = pd.DataFrame(report["metrics"]).T
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
        st.caption(f"Validation engine IDs: {report['test_engines']}")
else:
    st.header("Predict RUL for one observation")
    artifact_path = MODEL_DIR / "best_model.joblib"
    if not artifact_path.exists():
        st.info("No trained model found. Run `python -m predictive_maintenance.train --demo` first.")
    else:
        with st.form("prediction_form"):
            values = {}
            for feature in FEATURE_COLUMNS:
                values[feature] = st.number_input(feature, value=0.0, format="%.5f")
            submitted = st.form_submit_button("Estimate RUL")
        if submitted:
            try:
                artifact = load_artifact(artifact_path)
                result = predict_rul(artifact, pd.DataFrame([values]))[0]
                st.metric("Estimated remaining useful life", f"{result:.1f} cycles")
                if result <= 20:
                    st.error("High urgency estimate: investigate according to your maintenance procedure.")
                elif result <= 50:
                    st.warning("Elevated risk window: schedule engineering review.")
                else:
                    st.success("Lower estimated near-term risk; continue monitoring.")
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Prediction could not be completed: {error}")
