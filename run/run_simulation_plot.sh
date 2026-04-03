#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/src/simulation"
VENV_PYTHON="$MODEL_DIR/.venv/bin/python"

export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
export PYTHONPATH="$PROJECT_ROOT"

echo "[INFO] Setting up environment for $MODEL_DIR"
(
  cd "$MODEL_DIR"
  uv sync
)

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[ERROR] Virtual environment Python not found at $VENV_PYTHON" >&2
  exit 1
fi

echo "[INFO] Rendering simulation plots"
cd "$PROJECT_ROOT"
"$VENV_PYTHON" -m src.simulation.simulation_plot "$@"
