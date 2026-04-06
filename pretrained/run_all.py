from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_RUNNERS = {
    "xgboost": {
        "module": "pretrained.xgboost_runner",
        "venv": PROJECT_ROOT / "src" / "constrained_xgboost" / ".venv",
    },
    "catboost": {
        "module": "pretrained.catboost_runner",
        "venv": PROJECT_ROOT / "src" / "constrained_catboost" / ".venv",
    },
    "autogluon": {
        "module": "pretrained.autogluon_runner",
        "venv": PROJECT_ROOT / "src" / "automl" / ".venv",
    },
    "gravity": {
        "module": "pretrained.gravity_runner",
        "venv": PROJECT_ROOT / "src" / "gravity_model" / ".venv",
    },
}


def venv_python(venv_path: Path) -> str:
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def subprocess_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(project_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["UV_CACHE_DIR"] = str(project_root / ".uv-cache")
    return env


def sync_module_environment(module_dir: Path, env: dict[str, str]) -> None:
    print(f"Setting up environment for {module_dir}")
    subprocess.run(["uv", "sync"], check=True, cwd=module_dir, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate all predictions from pretrained artifacts.")
    parser.add_argument(
        "--overwrite-inputs",
        action="store_true",
        help="Recreate cached predictor input CSVs under pretrained/generated_inputs.",
    )
    parser.add_argument(
        "--skip-persist-inputs",
        action="store_true",
        help="Do not persist predictor input CSVs before scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    child_env = subprocess_env(PROJECT_ROOT)
    extra_args: list[str] = []
    if args.overwrite_inputs:
        extra_args.append("--overwrite-inputs")
    if args.skip_persist_inputs:
        extra_args.append("--skip-persist-inputs")

    for model_name, runner in MODEL_RUNNERS.items():
        print(f"Running {model_name} reproduction...")
        sync_module_environment(Path(runner["venv"]).parent, child_env)
        subprocess.run(
            [venv_python(runner["venv"]), "-m", runner["module"], *extra_args],
            check=True,
            cwd=PROJECT_ROOT,
            env=child_env,
        )


if __name__ == "__main__":
    main()
