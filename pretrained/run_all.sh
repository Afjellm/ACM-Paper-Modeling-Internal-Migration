#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SCRIPT_DIR/bootstrap.sh"

PROJECT_ROOT="$(get_project_root)"
set_pretrained_environment "$PROJECT_ROOT"

run_model() {
  MODULE_PATH="$1"
  RUNNER_MODULE="$2"
  shift 2
  sync_module_environment "$PROJECT_ROOT" "$MODULE_PATH"
  VENV_PYTHON="$(get_venv_python "$PROJECT_ROOT" "$MODULE_PATH")"
  "$VENV_PYTHON" -m "$RUNNER_MODULE" "$@"
}

run_model "src/constrained_xgboost" "pretrained.xgboost_runner" "$@"
run_model "src/constrained_catboost" "pretrained.catboost_runner" "$@"
run_model "src/gravity_model" "pretrained.gravity_runner" "$@"
run_model "src/automl" "pretrained.autogluon_runner" "$@"
