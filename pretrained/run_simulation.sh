#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SCRIPT_DIR/bootstrap.sh"

PROJECT_ROOT="$(get_project_root)"
set_pretrained_environment "$PROJECT_ROOT"
export MODEL_OUTPUT_BASE="models/output"
export ROWS_PER_AGE_GROUP="NONE"

sync_module_environment "$PROJECT_ROOT" "src/simulation"
sync_module_environment "$PROJECT_ROOT" "src/automl"
sync_module_environment "$PROJECT_ROOT" "src/constrained_xgboost"
sync_module_environment "$PROJECT_ROOT" "src/constrained_catboost"

VENV_PYTHON="$(get_venv_python "$PROJECT_ROOT" "src/simulation")"
echo "[INFO] Running simulation with pretrained artifacts from $MODEL_OUTPUT_BASE"
cd "$PROJECT_ROOT"
"$VENV_PYTHON" -m src.simulation.simulate_data "$@"
