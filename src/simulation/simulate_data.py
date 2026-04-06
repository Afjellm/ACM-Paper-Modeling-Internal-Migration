import os
import platform
from pathlib import Path

import pandas as pd
from src.data_loading.load_model_input import load_model_input_full
from src.model_artifacts import build_model_artifact_config, get_model_output_base
from src.simulation.gravity.load_models import discover_age_group_models, load_all_models
from src.simulation.gravity.simulate import simulate_data_with_gravity
import subprocess

AREA_CODES = [314,502]

YEARS_TO_SIMULATE = range(2018, 2024)

MODEL_ARTIFACTS = build_model_artifact_config()

GRAVITY_MODEL = MODEL_ARTIFACTS["gravity"]["model_name"]
GRAVITY_FIT_YEAR = MODEL_ARTIFACTS["gravity"]["fit_year"]
GRAVITY_MODEL_NEXT_YEAR = True

ENSEMBLE_MODEL = MODEL_ARTIFACTS["autogluon"]["model_name"]
ENSEMBLE_FIT_YEAR = MODEL_ARTIFACTS["autogluon"]["fit_year"]
ENSEMBLE_MODEL_NEXT_YEAR = False
ENSEMBLE_RELATIVE = False

XGBOOST_MODEL = MODEL_ARTIFACTS["xgboost"]["model_name"]
XGBOOST_FIT_YEAR = MODEL_ARTIFACTS["xgboost"]["fit_year"]
XGBOOST_NEXT_YEAR = True
XGBOOST_RELATIVE = True

CATBOOST_MODEL = MODEL_ARTIFACTS["constrained_catboost"]["model_name"]
CATBOOST_FIT_YEAR = MODEL_ARTIFACTS["constrained_catboost"]["fit_year"]
CATBOOST_NEXT_YEAR = False
CATBOOST_RELATIVE = True



simulation_files = ["distance.csv", "economy_down.csv", "economy_up.csv", "S1_S2_S3.csv", "university.csv",
                    "building.csv"]

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"

original_data = load_model_input_full(data_dir, use_age_groups_as_sub_dict=False)

for year, df in original_data.items():
    df["is_covid_year"] = year in [2020, 2021]
    df["is_covid_year"] = df["is_covid_year"].astype("category")
    df["income_ratio"] = df["gross_income_target"] / df["gross_income_origin"] * 100
    df["rent_ratio"] = df["rental_price_target"] / df["rental_price_origin"] * 100
    df["gpreis_ratio"] = df["land_price_target"] / df["land_price_origin"] * 100
    df["gdp_ratio"] = df["gdp_target"] / df["gdp_origin"] * 100
    df["immo_ratio"] = df["real_estate_price_target"] / df["real_estate_price_origin"] * 100
    df = df[df["area_code_origin"] != df["area_code_target"]]
    original_data[year] = df


combined_df = pd.concat(original_data.values(), ignore_index=True)
combined_df = combined_df[combined_df["area_code_origin"]!=combined_df["area_code_target"]]
na_rows = combined_df[combined_df.isna().any(axis=1)]

ensemble_predictions = {}
gravity_predictions = {}
xgboost_predictions = {}
catboost_predictions = {}

model_output_base = project_root / get_model_output_base()
base_path = model_output_base / "gravity_output" / GRAVITY_MODEL / GRAVITY_FIT_YEAR
model_paths = discover_age_group_models(base_path)

gravity_models = load_all_models(model_paths)

def venv_python(venv_path):
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def subprocess_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(project_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def trim_data_for_simulation(data, area_code):
    new_data = {}

    for year in data.keys():
        new_data[year] = data[year][(data[year]["area_code_target"] == area_code) |
                                    (data[year]["area_code_origin"] == area_code)].copy()

        new_data[year] = new_data[year][new_data[year]["area_code_origin"] != new_data[year]["area_code_target"]]

    return new_data


def _parse_rule_value(v: str):
    """Return ('delta' or 'set', float_value)."""
    s = str(v).strip()
    if s.startswith(('+', '-')):
        return 'delta', float(s)
    if s.startswith('*'):
        return 'factor', float(s.replace('*', ''))
    return 'set', float(s)


def apply_simulation_data(original_data, simulation_data, area_code):
    dfs_by_year = {year: df.copy() for year, df in original_data.items()}

    for _, r in simulation_data.iterrows():
        cols = [f"{r['column_name']}_origin", f"{r['column_name']}_target"]
        for y, df in dfs_by_year.items():
            y_str = str(y)
            if y_str not in r or pd.isna(r[y_str]):
                continue  # no instruction for this year

            mode, val = _parse_rule_value(r[y_str])

            # Ensure column exists
            if cols[0] not in df.columns or cols[1] not in df.columns:
                exit(f"column mismatch for {cols[0]} or {cols[1]}")

            # apply to ORIGIN side where area_code_origin == area_code
            mask_origin = df["area_code_origin"] == area_code

            # apply to TARGET side where area_code_target == area_code
            mask_target = df["area_code_target"] == area_code

            if mode == "delta":
                df.loc[mask_origin, cols[0]] = (
                        df.loc[mask_origin, cols[0]].astype(float) + val
                )
                df.loc[mask_target, cols[1]] = (
                        df.loc[mask_target, cols[1]].astype(float) + val
                )
            if mode == "factor":
                df.loc[mask_origin, cols[0]] = (
                        df.loc[mask_origin, cols[0]].astype(float) * val
                )
                df.loc[mask_target, cols[1]] = (
                        df.loc[mask_target, cols[1]].astype(float) * val
                )
            if mode == "set":
                df.loc[mask_origin, cols[0]] = float(val)
                df.loc[mask_target, cols[1]] = float(val)

    return dfs_by_year


def get_combined_simulation_results(simulation_data, area_code, file):
    xgboost_detailed = {}
    catboost_detailed = {}
    for year in YEARS_TO_SIMULATE:
        data_to_simulate = simulation_data[year].copy()
        data_to_simulate_prev_year = simulation_data[year-1].copy()

        if GRAVITY_MODEL_NEXT_YEAR:
            data_to_simulate_gravity = data_to_simulate_prev_year.copy()
        else:
            data_to_simulate_gravity = data_to_simulate.copy()

        os.makedirs("intermediate", exist_ok=True)

        ensemble_intermediate_output_path = f"intermediate/data_to_simulate_ensemble_{area_code}_{file}.csv"
        ensemble_intermediate_prev_output_path = f"intermediate/data_to_simulate_ensemble_prev_{area_code}_{file}.csv"

        xgboost_output_path = f"intermediate/xgboost_predictions_{area_code}_{file}.csv"
        xgboost_detailed_output_path = f"{xgboost_output_path.split('.')[0]}_{year}_full.csv"

        catboost_output_path = f"intermediate/catboost_predictions_{area_code}_{file}.csv"
        catboost_detailed_output_path = f"{catboost_output_path.split('.')[0]}_{year}_full.csv"
        ag_output_path = f"intermediate/ag_predictions_{area_code}_{file}.csv"

        data_to_simulate.to_csv(ensemble_intermediate_output_path, index=False)
        data_to_simulate_prev_year.to_csv(ensemble_intermediate_prev_output_path, index=False)

        xgboost_input_path = ensemble_intermediate_prev_output_path if XGBOOST_NEXT_YEAR else ensemble_intermediate_output_path
        catboost_input_path = ensemble_intermediate_prev_output_path if CATBOOST_NEXT_YEAR else ensemble_intermediate_output_path

        xgboost_ground_truth = "amount_next_year" if XGBOOST_NEXT_YEAR else "amount"
        catboost_ground_truth = "amount_next_year" if CATBOOST_NEXT_YEAR else "amount"

        project_root = Path(__file__).resolve().parents[2]
        ag_venv = project_root / "src" / "automl" / ".venv"
        catboost_venv = project_root / "src" / "constrained_catboost" / ".venv"
        xgb_venv = project_root / "src" / "constrained_xgboost" / ".venv"
        child_env = subprocess_env(project_root)

        subprocess.run([
            venv_python(ag_venv), str(project_root / "src" / "simulation" / "ensemble" / "simulate.py"),
            "--df", ensemble_intermediate_output_path,
            "--model-name", ENSEMBLE_MODEL,
            "--fit-year", ENSEMBLE_FIT_YEAR,
            "--ground-truth", "amount",
            "--area-code", str(area_code),
            "--out", ag_output_path,
            "--relative", str(ENSEMBLE_RELATIVE),
        ], check=True, cwd=project_root, env=child_env)

        subprocess.run([
            venv_python(xgb_venv), str(project_root / "src" / "simulation" / "xgboost_predictor" / "simulate.py"),
            "--df", xgboost_input_path,
            "--model-name", XGBOOST_MODEL,
            "--fit-year", XGBOOST_FIT_YEAR,
            "--ground-truth", xgboost_ground_truth,
            "--area-code", str(area_code),
            "--out", xgboost_output_path,
            "--detailed-output", xgboost_detailed_output_path,
            "--year", str(year),
            "--relative", str(XGBOOST_RELATIVE),
        ], check=True, cwd=project_root, env=child_env)

        subprocess.run([
            venv_python(catboost_venv), str(project_root / "src" / "simulation" / "catboost_predictor" / "simulate.py"),
            "--df", catboost_input_path,
            "--model-name", CATBOOST_MODEL,
            "--fit-year", CATBOOST_FIT_YEAR,
            "--ground-truth", catboost_ground_truth,
            "--area-code", str(area_code),
            "--out", catboost_output_path,
            "--detailed-output", catboost_detailed_output_path,
            "--year", str(year),
            "--relative", str(CATBOOST_RELATIVE),
        ], check=True, cwd=project_root, env=child_env)

        xgboost_predictions[year] = pd.read_csv(xgboost_output_path)
        catboost_predictions[year] = pd.read_csv(catboost_output_path)
        ensemble_predictions[year] = pd.read_csv(ag_output_path)
        xgboost_detailed[year] = pd.read_csv(xgboost_detailed_output_path)
        catboost_detailed[year] = pd.read_csv(catboost_detailed_output_path)

        gravity_predictions[year] = simulate_data_with_gravity(data_to_simulate_gravity, gravity_models,
                                                               'amount_next_year', area_code, year)

        os.remove(catboost_output_path)
        os.remove(catboost_detailed_output_path)
        os.remove(xgboost_detailed_output_path)
        os.remove(ag_output_path)
        os.remove(xgboost_output_path)

    rows = []

    xgboost_detailed_output = pd.concat([xgboost_detailed[year] for year in YEARS_TO_SIMULATE], axis=0, ignore_index=True)
    catboost_detailed_output = pd.concat([catboost_detailed[year] for year in YEARS_TO_SIMULATE], axis=0, ignore_index=True)
    os.makedirs("intermediate/xgboost_detailed", exist_ok=True)
    os.makedirs("intermediate/catboost_detailed", exist_ok=True)
    xgboost_detailed_output.to_csv(f"intermediate/xgboost_detailed/xgboost_predictions_{area_code}_{file}.csv", index=False)
    catboost_detailed_output.to_csv(f"intermediate/catboost_detailed/catboost_predictions_{area_code}_{file}.csv", index=False)

    for y in YEARS_TO_SIMULATE:
        e_df = ensemble_predictions[y]
        g_df = gravity_predictions[y]
        x_df = xgboost_predictions[y]
        c_df = catboost_predictions[y]

        # Basic sanity checks
        required_cols = {"direction", "predicted", "amount"}
        if set(e_df.columns) & required_cols != required_cols:
            raise ValueError(f"Ensemble df for {y} is missing required columns.")
        if set(g_df.columns) & required_cols != required_cols:
            raise ValueError(f"Gravity df for {y} is missing required columns.")
        if set(x_df.columns) & required_cols != required_cols:
            raise ValueError(f"XGBoost df for {y} is missing required columns.")
        if set(c_df.columns) & required_cols != required_cols:
            raise ValueError(f"XGBoost df for {y} is missing required columns.")

        # Ensure both have exactly 'inward' and 'outward'
        e_idx = e_df.set_index("direction")
        g_idx = g_df.set_index("direction")
        x_idx = x_df.set_index("direction")
        c_idx = c_df.set_index("direction")

        for direction in ["inward", "outward"]:
            if direction not in e_idx.index or direction not in g_idx.index or direction not in x_idx.index or direction not in c_idx.index:
                raise ValueError(f"Missing direction '{direction}' in year {y}.")

        # Assert amounts match between ensemble and gravity for both directions
        amount_in_e = e_idx.loc["inward", "amount"]
        amount_out_e = e_idx.loc["outward", "amount"]
        amount_in_g = g_idx.loc["inward", "amount"]
        amount_out_g = g_idx.loc["outward", "amount"]
        amount_in_c = c_idx.loc["inward", "amount"]

        assert amount_in_e == amount_in_g, f"Amount_in mismatch in {y}: ensemble={amount_in_e}, gravity={amount_in_g}"
        assert amount_out_e == amount_out_g, f"Amount_out mismatch in {y}: ensemble={amount_out_e}, gravity={amount_out_g}"
        assert amount_in_c == amount_in_g, f"Amount_in mismatch in {y}: ensemble={amount_in_c}, gravity={amount_in_g}"

        # Build the combined row
        row = {
            "year": y,
            "amount_in": amount_in_e,
            "amount_out": amount_out_e,
            "gravity_predicted_in": g_idx.loc["inward", "predicted"],
            "gravity_predicted_out": g_idx.loc["outward", "predicted"],
            "ensemble_in": e_idx.loc["inward", "predicted"],
            "ensemble_out": e_idx.loc["outward", "predicted"],
            "xg_in": x_idx.loc["inward", "predicted"],
            "xg_out": x_idx.loc["outward", "predicted"],
            "cb_in": c_idx.loc["inward", "predicted"],
            "cb_out": c_idx.loc["outward", "predicted"],
        }
        rows.append(row)

    combined = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    combined["delta_gravity"] = combined["gravity_predicted_in"] - combined["gravity_predicted_out"]
    combined["delta_ensemble"] = combined["ensemble_in"] - combined["ensemble_out"]
    combined["delta_xg"] = combined["xg_in"] - combined["xg_out"]
    combined["delta_cb"] = combined["cb_in"] - combined["cb_out"]
    combined["delta_amount"] = combined["amount_in"] - combined["amount_out"]

    return combined


all_results = []

for area_code in AREA_CODES:
    data_trimmed = trim_data_for_simulation(original_data, area_code)

    original_predictions = get_combined_simulation_results(data_trimmed, area_code, "base")
    original_predictions["area_code"] = area_code
    original_predictions["scenario_name"] = "original"
    all_results.append(original_predictions)

    for file in simulation_files:
        print("Processing", file)
        file_path = project_root / "src" / "simulation" / "simulation_data" / file
        simulation_delta = pd.read_csv(file_path, delimiter=';', dtype=str)
        simulation_data = apply_simulation_data(data_trimmed, simulation_delta, area_code)

        scenario_name = file.replace(".csv", "")
        simulation_predictions = get_combined_simulation_results(simulation_data, area_code, scenario_name)

        simulation_predictions["area_code"] = area_code
        simulation_predictions["scenario_name"] = scenario_name
        all_results.append(simulation_predictions)

final_df = pd.concat(all_results, ignore_index=True)
output_path = project_root / "src" / "simulation" / "output"
os.makedirs(output_path, exist_ok=True)
final_df.to_csv(output_path / "paper_simulation.csv", index=False)
print(f"✅ Combined CSV saved to {output_path}/paper_simulation.csv")
