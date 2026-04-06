#!/usr/bin/env sh
set -eu

get_project_root() {
  CDPATH= cd -- "$(dirname -- "$0")/.." && pwd
}

set_pretrained_environment() {
  PROJECT_ROOT="$1"
  export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
  if [ "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
  else
    export PYTHONPATH="$PROJECT_ROOT"
  fi
}

sync_module_environment() {
  PROJECT_ROOT="$1"
  RELATIVE_MODULE_PATH="$2"
  MODULE_DIR="$PROJECT_ROOT/$RELATIVE_MODULE_PATH"
  echo "[INFO] Setting up environment for $MODULE_DIR"
  (
    cd "$MODULE_DIR"
    uv sync
  )
}

get_venv_python() {
  PROJECT_ROOT="$1"
  RELATIVE_MODULE_PATH="$2"
  UNIX_VENV_PYTHON="$PROJECT_ROOT/$RELATIVE_MODULE_PATH/.venv/bin/python"
  WINDOWS_VENV_PYTHON="$PROJECT_ROOT/$RELATIVE_MODULE_PATH/.venv/Scripts/python.exe"

  if [ -x "$UNIX_VENV_PYTHON" ]; then
    printf '%s\n' "$UNIX_VENV_PYTHON"
    return
  fi

  if [ -f "$WINDOWS_VENV_PYTHON" ]; then
    printf '%s\n' "$WINDOWS_VENV_PYTHON"
    return
  fi

  echo "[ERROR] Virtual environment Python not found at $UNIX_VENV_PYTHON or $WINDOWS_VENV_PYTHON" >&2
  exit 1
}
