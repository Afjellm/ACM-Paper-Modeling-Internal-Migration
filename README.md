# Modeling and Simulation of Internal Migration using Machine Learning Regression and the Gravity Model of Migration

Code and data artifacts for the paper [ACM_paper.pdf](./ACM_paper.pdf) for SIGSIM PADS 2026.

## Authors

Corresponding Author: <br />
Alexander Jell - e51820065@student.tuwien.ac.at

Authors:
- Daniele Giannandrea - daniele.giannandrea@tuwien.ac.at
- Niki Popper - nikolas.popper@tuwien.ac.at
- Martin Bicher - martin.bicher@tuwien.ac.at


## Introduction

This repository contains the data preparation outputs, training code, analysis scripts, and simulation code used for the
paper.

### Where to find what

- [`data/`](./data) contains the yearly input data used by the models. The dataset documentation lives in [
  `data/Readme.md`](./data/Readme.md).
- [`run/`](./run) contains the entry-point scripts for Windows (`.bat`) and Linux (`.sh`).
- [`src/automl/`](./src/automl) contains the AutoGluon training code.
- [`src/constrained_xgboost/`](./src/constrained_xgboost) contains the constrained XGBoost training code.
- [`src/constrained_catboost/`](./src/constrained_catboost) contains the constrained CatBoost training code.
- [`src/gravity_model/`](./src/gravity_model) contains the gravity model fitting code.
- [`src/analysis/`](./src/analysis) contains the scripts that compare model outputs and generate the analysis report.
- [`src/simulation/`](./src/simulation) contains the code and scenario inputs for simulation runs.
- [`models/`](./models) is expected to contain trained model artifacts after local training runs.
- [`intermediate/`](./intermediate) stores temporary files created during simulation runs.

## Installation

The runner scripts use [`uv`](https://docs.astral.sh/uv/) to create per-module virtual environments and install
dependencies from the `pyproject.toml` and `uv.lock` files in each subproject.

### Install `uv`

Official installation instructions are available in the `uv`
documentation: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your shell and verify:

```powershell
uv --version
```

#### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your shell and verify:

```bash
uv --version
```

### Python

If Python is not already available on your system, `uv` can install it for you:

```bash
uv python install
```

## How to run

All runner scripts live in [`run/`](./run). They do two things for you:

1. call `uv sync` in the corresponding module directory
2. run the correct Python entry point with the repository root on `PYTHONPATH`

### Step 1: Configure `run/settings.env`

Before running the training, analysis, or simulation scripts, review [`run/settings.env`](./run/settings.env):

For fast execution to simply run the models end to end the following settings can be used:

```env
ROWS_PER_AGE_GROUP=1000
TIME_LIMIT=120
```

- `ROWS_PER_AGE_GROUP` limits how many rows are loaded per age group. Set it to `NONE` to use the full dataset.
  The same value is also used to derive the expected model artifact names for `run_analysis` and
  `run_simulate_data`. For example, `ROWS_PER_AGE_GROUP=1000` maps to model names ending in `rows_1000`, while
  `ROWS_PER_AGE_GROUP=NONE` maps to model names ending in `full`.
- `TIME_LIMIT` is used by the AutoML training script and is interpreted in seconds.

If you change `ROWS_PER_AGE_GROUP`, retrain all four model families with the same setting before running analysis or
simulation. Otherwise those scripts will look for a different model folder name than the one written during training.

To numerically reproduce the paper results the following settings are required:

```env
ROWS_PER_AGE_GROUP=NONE
TIME_LIMIT=14400
```

### Step 2: Run the scripts

From the repository root, use the launchers in [`run/`](./run).

The scripts ``run_analysis`` and ``run_simulate_data`` are only runnable after all 4 model outputs have been generated.
They also read [`run/settings.env`](./run/settings.env), so the configured `ROWS_PER_AGE_GROUP` must match the training
run whose artifacts you want to analyze or simulate.

#### Windows

```powershell
.\run\run_automl.bat
.\run\run_constrained_xgboost.bat
.\run\run_constrained_catboost.bat
.\run\run_gravity.bat
.\run\run_analysis.bat
.\run\run_simulate_data.bat
```

#### Linux

```bash
./run/run_automl.sh
./run/run_constrained_xgboost.sh
./run/run_constrained_catboost.sh
./run/run_gravity.sh
./run/run_analysis.sh
./run/run_simulate_data.sh
```

## I want to reproduce the paper results

Train the models first, then run the analysis script.

Expected duration on full data:
- gravity model fitting: 1h
- xgboost model training: 2h
- catboost model training: 3h
- autogluon model training: 4h
- simulation of scenarios: 20m

Specs of the author that ran experiments:

RAM: 64 GB <br />
CPU: Intel64 Family 6 Model 183 Stepping 1 GenuineIntel ~3500 Mhz <br />
OS Name: Microsoft Windows 11 <br />
OS Version: 10.0.26200 <br />


Recommended order:

1. configure [`run/settings.env`](./run/settings.env)
2. run AutoML, constrained XGBoost, constrained CatBoost, and gravity training
3. run the analysis script to generate the comparison report
4. run the simulation script to generate data for the simulation plots

Training outputs are written under `models/output/`, for example:

- `models/output/autogluon_output/`
- `models/output/xgboost_output/`
- `models/output/catboost_output/`
- `models/output/gravity_output/`

The comparison script reads these artifacts and generates the model comparison report. The exact artifact name it reads
is derived automatically from [`run/settings.env`](./run/settings.env).

### Why are the results not 100% identical

Running the models in this repository with the above provided settings will reproduce the results presented in the paper
almost identical. The difference
in the most important evaluation metric ``MARE for in migration`` only differs by 1%. Differences in the number of
correctly predicted districts also only slightly differs:

| model     | mare_in_repo | mare_in_paper (rounded) |
|-----------|--------------|-------------------------|
| xgboost   | 0.139802     | 0.14                    |
| catboost  | 0.262517     | 0.25                    |
| autogluon | 0.089937     | 0.09                    |
| gravity   | 0.266472     | 0.25                    |

| model     | correctly predicted (S_agree) repo | correctly predicted (paper) |
|-----------|------------------------------------|-----------------------------|
| xgboost   | 88/116                             | 90/116                      |
| catboost  | 95/116                             | 98/116                      |
| autogluon | 96/116                             | 97/116                      |
| gravity   | 88/116                             | 86/116                      |

Reported R2 scores also only differ by at max 0.02. 

Some discrepancies between the results presented in the paper and those obtained from this repository arise from data that cannot be shared publicly. In particular, tourism-related data used in the original experiments were provided externally and are not included in this repository.

The primary objective of this study is not to achieve or surpass benchmark performance on public datasets. Instead, the focus is on analyzing the strengths and limitations of different modeling approaches, as well as identifying key challenges in modeling internal migration in Austria.

Importantly, the absence of the non-public data leads only to minor numerical differences and does not affect the overall conclusions. The core insights regarding model behavior, comparative advantages, and modeling challenges remain consistent.
## I want to reuse the artifacts

The trained models can be reused for new data. The process to use these models with replacement data is as follows.

### 1. Replace the data in `data/`

Replace the files in the yearly folders inside [`data/`](./data). The filenames have to stay identical:

- [`migration.csv`](./data/2023/migration.csv)
- [`metadata.csv`](./data/2023/metadata.csv)
- [`distance_matrix.csv`](./data/2023/distance_matrix.csv)

The shared loader expects this exact structure and reads those files for each year from [
`src/data_loading/load_model_input.py`](./src/data_loading/load_model_input.py).

### Distance matrix

For the distance matrix file and migration file, the same join column names are needed.

The distance matrix file expects:

- `area_code_origin`
- `area_code_target`

The file should contain mirrored rows. For example, if the row `101 -> 105` exists, then `105 -> 101` also has to exist.
In the current setup the values are usually the same in both directions, but in principle they can differ.

### Migration

The migration file is expected to contain the migration ground truth.

- `bezirk_origin` and `bezirk_target` are currently expected by the loader, even though they are not needed by the
  trained models themselves.
- `area_code_origin` and `area_code_target` are required for joins.
- `amount`, `amount_prev_year`, and `amount_next_year` denote the ground-truth migration data.
- Anything else is expected with the same name if downstream code depends on it.

### Metadata

The metadata file must contain:

- `area_code` as the join key
- `time` as the time column

Every other metadata column can be replaced with new metadata columns and can be changed, but then the training code has
to be updated accordingly.

The assembled input schema is defined in:

- [`src/data_loading/load_model_input.py`](./src/data_loading/load_model_input.py)

The column selections used during training are defined in:

- [`src/automl/autogluon_training.py`](./src/automl/autogluon_training.py)
- [`src/constrained_xgboost/xgboost_training.py`](./src/constrained_xgboost/xgboost_training.py)
- [`src/constrained_catboost/catboost_model.py`](./src/constrained_catboost/catboost_model.py)
- [`src/gravity_model/gravity_model.py`](./src/gravity_model/gravity_model.py)

If you replace metadata columns with different ones, these are the sections that need to be adapted before retraining or
reusing the pipeline.
