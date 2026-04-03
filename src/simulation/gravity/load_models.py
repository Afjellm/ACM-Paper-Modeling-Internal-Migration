
import sys
from pathlib import Path
from typing import Dict, Optional, Union
import statsmodels.api as sm

MODEL_EXTS = (".pkl", ".pickle")


def _find_first_year_dir(span_dir: Path) -> Path:
    """Return the subfolder for the *first* year inside a span folder (e.g., 2018 in 2018_2019)."""
    year_dirs = []
    for d in span_dir.iterdir():
        if d.is_dir():
            name = d.name.strip()
            if name.isdigit():
                year_dirs.append((int(name), d))
    if not year_dirs:
        raise FileNotFoundError(f"No numeric year subfolders found under {span_dir}")
    year_dirs.sort(key=lambda x: x[0])
    return year_dirs[0][1]


def _pick_model_file(age_dir: Path) -> Optional[Path]:
    """Pick the newest pickle file inside an age-group folder, or None if not found."""
    cands = [p for p in age_dir.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXTS]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def discover_age_group_models(base_dir: Union[str, Path], year_span: Optional[str] = None) -> Dict[str, Path]:
    """Discover model pickle paths for all age groups under the FIRST year of a span.

    If `year_span` is None, the lexicographically smallest span folder is used.

    Returns a dict mapping age-group label (folder name) -> pickle path.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    if year_span is None:
        span_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        if not span_dirs:
            raise FileNotFoundError(f"No year-span folders found under {base}")
        span_dir = span_dirs[0]
    else:
        span_dir = base / year_span
        if not span_dir.exists():
            raise FileNotFoundError(f"Year-span folder not found: {span_dir}")

    models: Dict[str, Path] = {}
    for age_dir in sorted([d for d in span_dir.iterdir() if d.is_dir()]):
        model_path = _pick_model_file(age_dir)
        if model_path is None:
            print(f"[WARN] No pickle found in {age_dir}", file=sys.stderr)
            continue
        models[age_dir.name] = model_path

    return models


def load_result_model(pkl_path: Union[str, Path]):
    """Load a statsmodels results object from pickle.

    Tries `statsmodels.api.load` first (compatible with `.save()`), then raw pickle.
    """
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"Model pickle not found: {p}")

    try:
        return sm.load(str(p))
    except Exception:
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)


def load_all_models(model_paths: Dict[str, Path]) -> Dict[str, object]:
    """Load all models given a mapping of age-group -> pickle path."""
    loaded = {}
    for age, path in model_paths.items():
        try:
            loaded[age] = load_result_model(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {path} for age '{age}': {e}", file=sys.stderr)
    if not loaded:
        raise RuntimeError("No models could be loaded.")
    return loaded