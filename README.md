# Predictive Maintenance and Remaining Useful Life (RUL) Prediction

An end-to-end, portfolio-ready machine-learning project for estimating the **Remaining Useful Life (RUL)** of turbofan engines. The implementation uses the public NASA C-MAPSS FD001 dataset when available and also includes a clearly labelled synthetic demo mode so the pipeline can be tested without downloading data.

## What this project demonstrates

- RUL **regression**: predict how many operational cycles remain before failure.
- Leakage-safe splitting by engine unit (never split individual rows from one engine across train and test).
- Training-only preprocessing and reproducible model artifacts.
- A simple, explainable linear baseline compared with tree-based models.
- Evaluation with MAE and RMSE, plus an interactive Streamlit dashboard.

Failure classification and RUL regression are different problems. Classification answers “will this engine fail within a chosen horizon?” and uses labels such as `0/1`. RUL regression predicts a continuous time-to-failure value and preserves more information about the maintenance horizon. This project focuses on regression; a classification extension is described under Future improvements.

## Dataset

This project targets **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**, preferably subset FD001:

- Source: [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- FD001 contains one operating condition and one fault mode, which makes it a good beginner-friendly benchmark.
- Each row represents one engine at one operating cycle. The 3 operating settings and 21 sensor channels are followed by `unit_id` and `cycle` in the raw text files used by this project.
- For each training engine, the target is `max(cycle for engine) - current_cycle`, optionally capped at 125 cycles (the common C-MAPSS piecewise-linear target).

Download `CMAPSSData.zip` from NASA, extract it, and place `train_FD001.txt` in `data/raw/`. The loader also accepts `test_FD001.txt` and `RUL_FD001.txt`, but the default training workflow uses a leakage-safe split of the labelled training engines. NASA data is simulated rather than collected fleet telemetry; results should not be treated as production reliability estimates. FD001 has a single condition/fault mode, and domain shift, censoring, maintenance actions, and sensor calibration are not represented.

## Architecture

```text
data/raw/                 # local NASA files (ignored by git)
data/processed/           # generated feature tables (ignored by git)
models/                   # generated model artifacts (ignored by git)
reports/                  # generated metrics and plots (ignored by git)
src/predictive_maintenance/
  data.py                 # loading, validation, target creation, demo data
  features.py             # feature selection and preprocessing
  models.py               # model definitions and persistence
  train.py                # CLI training/evaluation entry point
  predict.py              # artifact loading and inference
  eda.py                  # exploratory plots
  config.py               # paths and constants
app.py                    # Streamlit dashboard
tests/                    # focused unit tests
```

## Windows installation

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal, or invoke `.venv\Scripts\python.exe` directly.

## Prepare data and train

After extracting NASA's archive:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Copy-Item "$env:USERPROFILE\Downloads\CMAPSSData\train_FD001.txt" data\raw\train_FD001.txt
.\.venv\Scripts\python.exe -m predictive_maintenance.train --data data\raw\train_FD001.txt
```

To validate the complete workflow without NASA data (this is only a pipeline smoke test, not a NASA result):

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance.train --demo
```

The command writes `models/best_model.joblib`, `models/preprocessor.joblib`, and `reports/metrics.json`. It prints actual validation metrics; no metrics are hard-coded in this repository. Generate EDA plots with:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance.eda --data data\raw\train_FD001.txt
```

### Verified demo smoke-test result

On 2026-10-14, the deterministic `--demo` workflow completed successfully using 12 synthetic engines. Its engine-level holdout contained engines 1 and 8, and the observed metrics were:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Linear regression | 4.354 | 5.647 |
| Random forest | 3.014 | 5.039 |
| Histogram gradient boosting | 2.736 | 4.407 |

These are pipeline verification metrics only, not NASA C-MAPSS results and not evidence of production performance. NASA metrics will be generated after you download the dataset and run the real-data command.

## Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard shows project and dataset information, available evaluation metrics, sensor distributions, and a prediction form. Missing artifacts and invalid inputs are displayed as actionable warnings rather than causing a blank page. A prediction is an estimate, not a maintenance release decision; use engineering controls, inspections, and validated fleet data before acting on it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover C-MAPSS parsing, target generation, engine-level splitting, preprocessing, persistence, and inference. They use small in-memory data and do not require NASA files.

## Example usage

```python
from predictive_maintenance.data import load_cmapss, add_rul_target
from predictive_maintenance.models import load_artifact
from predictive_maintenance.predict import predict_rul

frame = add_rul_target(load_cmapss("data/raw/train_FD001.txt"))
bundle = load_artifact("models/best_model.joblib")
prediction = predict_rul(bundle, frame.drop(columns=["rul"]).tail(1))
print(prediction)
```

## Limitations and future improvements

The benchmark is simulated, uses only FD001 in the documented path, and does not model maintenance interventions or true fleet censoring. Future work should validate on FD002–FD004, add time-window and trend features, use temporal cross-validation, quantify uncertainty, monitor drift, calibrate a failure-horizon classifier, and establish safety/maintenance governance.

## GitHub publishing

Do not commit `data/raw/`, generated models, reports, virtual environments, or credentials. Review `git status`, add only source, tests, documentation, and configuration examples, then:

```powershell
git init
git add README.md requirements.txt .gitignore src app.py tests config.example.yaml
git commit -m "Build predictive maintenance RUL pipeline`n`nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```
