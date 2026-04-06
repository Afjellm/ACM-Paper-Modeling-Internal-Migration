# Pretrained Reproduction steps

## 1. Download the models

The pretrained models can be downloaded under: https://doi.org/10.5281/zenodo.19437425

Unzip the folder into a models folder in the repository root. So the structure after unzipping should be:

```
<repository root>
- models
  - output
    - autogluon_output
    - catboost_output
    - gravity_output
    - xgboost_output
```

## 2. Run all models

The wrappers in `pretrained/` now bootstrap their own environments. Each script runs `uv sync` in the required module
before executing the matching runner, so you do not need to create the `.venv` folders manually.

Depending on your os:

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\pretrained\run_all.ps1
```

```bash
./pretrained/run_all.sh
```

Optional flags:

```powershell
powershell -ExecutionPolicy Bypass -File .\pretrained\run_all.ps1 --overwrite-inputs
powershell -ExecutionPolicy Bypass -File .\pretrained\run_all.ps1 --skip-persist-inputs
```

## Run a single model

```powershell
powershell -ExecutionPolicy Bypass -File .\pretrained\run_xgboost.ps1
powershell -ExecutionPolicy Bypass -File .\pretrained\run_catboost.ps1
powershell -ExecutionPolicy Bypass -File .\pretrained\run_autogluon.ps1
powershell -ExecutionPolicy Bypass -File .\pretrained\run_gravity.ps1
```

`pretrained/run_all.py` is also available as a Python entry point. It now performs the same `uv sync` setup before
dispatching each runner into the corresponding project virtual environment.

## 3. Analyze the recreated outputs

Model outputs can be analyzed in the following way:

```powershell
$env:MODEL_OUTPUT_BASE = "pretrained/output"
python -m src.analysis.analyze_output --output pretrained/model_comparison_analysis.md
```

## 4. Run simulation with the pretrained models

The simulation wrappers use the downloaded pretrained artifacts from `models/output`, which is the location expected
after unpacking the Zenodo archive into the repository root.

Use the dedicated wrappers from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\pretrained\run_simulation.ps1
powershell -ExecutionPolicy Bypass -File .\pretrained\run_simulation_plot.ps1
```

```bash
./pretrained/run_simulation.sh
./pretrained/run_simulation_plot.sh
```

The simulation wrappers automatically:

Outputs are written to:

- `src/simulation/output/paper_simulation.csv`
- `src/simulation/output/simulation_plot.svg` (or the path passed via `--output`)

## Notes

- XGBoost and Gravity use the previous year's input matrix because their configured target is `amount_next_year`.
- CatBoost and AutoGluon score the same year directly because their configured target is `amount`.
