#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. "$SCRIPT_DIR/bootstrap.sh"

PROJECT_ROOT="$(get_project_root)"
set_pretrained_environment "$PROJECT_ROOT"
sync_module_environment "$PROJECT_ROOT" "src/simulation"

VENV_PYTHON="$(get_venv_python "$PROJECT_ROOT" "src/simulation")"
echo "[INFO] Rendering simulation plots"
cd "$PROJECT_ROOT"
"$VENV_PYTHON" -m src.simulation.simulation_plot "$@"
