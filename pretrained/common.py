from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data_loading.col_normalize import norm_count_cols
from src.data_loading.load_model_input import load_model_input_full
from src.model_artifacts import build_model_artifact_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
GENERATED_INPUTS_DIR = PRETRAINED_DIR / "generated_inputs"
PRETRAINED_OUTPUT_ROOT = PRETRAINED_DIR / "output"


MODEL_SPECS: dict[str, dict[str, object]] = {
    "xgboost": {
        "artifact_key": "xgboost",
        "output_root_name": "xgboost_output",
        "ground_truth_column": "amount_next_year",
        "use_previous_year_input": True,
        "relative": True,
        "per_age_group_output": True,
    },
    "catboost": {
        "artifact_key": "constrained_catboost",
        "output_root_name": "catboost_output",
        "ground_truth_column": "amount",
        "use_previous_year_input": False,
        "relative": True,
        "per_age_group_output": True,
    },
    "autogluon": {
        "artifact_key": "autogluon",
        "output_root_name": "autogluon_output",
        "ground_truth_column": "amount",
        "use_previous_year_input": False,
        "relative": False,
        "per_age_group_output": False,
    },
    "gravity": {
        "artifact_key": "gravity",
        "output_root_name": "gravity_output",
        "ground_truth_column": "amount_next_year",
        "use_previous_year_input": True,
        "relative": False,
        "per_age_group_output": True,
    },
}


def _add_derived_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    enriched = df.copy()
    enriched["is_covid_year"] = (year in [2020, 2021])
    enriched["is_covid_year"] = enriched["is_covid_year"].astype("category")
    enriched["income_ratio"] = enriched["gross_income_target"] / enriched["gross_income_origin"] * 100
    enriched["rent_ratio"] = enriched["rental_price_target"] / enriched["rental_price_origin"] * 100
    enriched["gpreis_ratio"] = enriched["land_price_target"] / enriched["land_price_origin"] * 100
    enriched["gdp_ratio"] = enriched["gdp_target"] / enriched["gdp_origin"] * 100
    enriched["immo_ratio"] = enriched["real_estate_price_target"] / enriched["real_estate_price_origin"] * 100
    return enriched


def load_prepared_input_frames(data_dir: Path = DATA_DIR) -> dict[int, pd.DataFrame]:
    prepared = {}
    loaded = load_model_input_full(str(data_dir), use_age_groups_as_sub_dict=False)
    for year, df in loaded.items():
        prepared[year] = _add_derived_columns(df, year)
    return prepared


def ensure_predictor_inputs(
    prepared_data: dict[int, pd.DataFrame],
    base_dir: Path = GENERATED_INPUTS_DIR,
    overwrite: bool = False,
) -> dict[int, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    persisted_paths: dict[int, Path] = {}

    for year, df in sorted(prepared_data.items()):
        year_dir = base_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        csv_path = year_dir / "predictor_input_full.csv"
        if overwrite or not csv_path.exists():
            df.to_csv(csv_path, index=False)
        persisted_paths[year] = csv_path

    return persisted_paths


def get_model_spec(model_name: str) -> dict[str, object]:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model '{model_name}'.") from exc


def get_model_artifact_details(model_name: str) -> dict[str, str]:
    spec = get_model_spec(model_name)
    config = build_model_artifact_config()[str(spec["artifact_key"])]
    source_root = PROJECT_ROOT / "models" / "output" / str(spec["output_root_name"])
    output_root = PRETRAINED_OUTPUT_ROOT / str(spec["output_root_name"])
    return {
        "model_name": config["model_name"],
        "fit_year": config["fit_year"],
        "source_dir": str(source_root / config["model_name"] / config["fit_year"]),
        "output_dir": str(output_root / config["model_name"] / config["fit_year"]),
        "model_root": str(source_root),
    }


def get_prediction_years(
    prepared_data: dict[int, pd.DataFrame],
    model_name: str,
) -> list[int]:
    spec = get_model_spec(model_name)
    available_years = sorted(prepared_data)
    if not available_years:
        return []

    if bool(spec["use_previous_year_input"]):
        return [year for year in available_years if (year - 1) in prepared_data]
    return available_years


def get_input_frame_for_prediction(
    prepared_data: dict[int, pd.DataFrame],
    model_name: str,
    prediction_year: int,
) -> pd.DataFrame:
    spec = get_model_spec(model_name)
    input_year = prediction_year - 1 if bool(spec["use_previous_year_input"]) else prediction_year
    return prepared_data[input_year].copy()


def prepare_model_frame(df: pd.DataFrame, relative: bool) -> pd.DataFrame:
    prepared = df.copy()
    if relative:
        prepared = norm_count_cols(prepared)

    prepared["is_adjacent"] = prepared["is_adjacent"].astype("category")
    prepared["is_covid_year"] = prepared["is_covid_year"].astype("category")
    return prepared


def create_aggregated_migrations_from_predicted_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out_migrations = (
        df.groupby(["area_code_origin"])[["amount", "predicted"]]
        .sum()
        .reset_index()
        .rename(columns={"amount": "amount_out", "predicted": "predicted_out"})
    )

    in_migrations = (
        df.groupby(["area_code_target"])[["amount", "predicted"]]
        .sum()
        .reset_index()
        .rename(columns={"amount": "amount_in", "predicted": "predicted_in"})
    )

    aggregated = pd.merge(
        in_migrations,
        out_migrations,
        how="outer",
        left_on=["area_code_target"],
        right_on=["area_code_origin"],
    ).fillna(0)

    aggregated["Relative Error Out"] = aggregated.apply(
        lambda row: abs(row["amount_out"] - row["predicted_out"]) / row["amount_out"]
        if row["amount_out"] != 0
        else 0,
        axis=1,
    )
    aggregated["Relative Error In"] = aggregated.apply(
        lambda row: abs(row["amount_in"] - row["predicted_in"]) / row["amount_in"]
        if row["amount_in"] != 0
        else 0,
        axis=1,
    )

    return aggregated


def write_prediction_outputs(
    *,
    model_name: str,
    prediction_year: int,
    fit_year: str,
    full_predictions: pd.DataFrame,
) -> None:
    details = get_model_artifact_details(model_name)
    spec = get_model_spec(model_name)
    base_output_dir = Path(details["output_dir"]) / str(prediction_year)

    if bool(spec["per_age_group_output"]):
        for age_group, age_group_results in full_predictions.groupby("age_group"):
            age_dir = base_output_dir / str(age_group)
            age_dir.mkdir(parents=True, exist_ok=True)
            _write_single_prediction_output(age_group_results, age_dir)
    else:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        _write_single_prediction_output(full_predictions, base_output_dir)


def _write_single_prediction_output(predictions: pd.DataFrame, output_dir: Path) -> None:
    predictions = predictions.copy()
    with_within = create_aggregated_migrations_from_predicted_dataframe(predictions)
    without_within_rows = predictions.loc[
        predictions["area_code_origin"] != predictions["area_code_target"]
    ].copy()
    without_within = create_aggregated_migrations_from_predicted_dataframe(without_within_rows)

    predictions.to_csv(output_dir / "full_results.csv", index=False)
    with_within.to_csv(output_dir / "results_with_within_migration.csv", index=False)
    without_within.to_csv(output_dir / "results_without_within_migration.csv", index=False)


def run_and_write_predictions(
    *,
    model_name: str,
    predictor: Callable[[pd.DataFrame, int], pd.DataFrame],
    persist_inputs: bool = True,
    overwrite_inputs: bool = False,
) -> list[Path]:
    prepared_data = load_prepared_input_frames()
    if persist_inputs:
        ensure_predictor_inputs(prepared_data, overwrite=overwrite_inputs)

    details = get_model_artifact_details(model_name)
    written_paths: list[Path] = []

    for prediction_year in get_prediction_years(prepared_data, model_name):
        frame = get_input_frame_for_prediction(prepared_data, model_name, prediction_year)
        predictions = predictor(frame, prediction_year)
        write_prediction_outputs(
            model_name=model_name,
            prediction_year=prediction_year,
            fit_year=details["fit_year"],
            full_predictions=predictions,
        )
        written_paths.append(Path(details["output_dir"]) / str(prediction_year))

    return written_paths


def print_written_paths(model_name: str, paths: list[Path]) -> None:
    print(f"Wrote {model_name} predictions for {len(paths)} year(s):")
    for path in paths:
        print(f" - {path}")
