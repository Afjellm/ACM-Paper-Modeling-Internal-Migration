from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pretrained.common import (
    get_model_artifact_details,
    get_model_spec,
    print_written_paths,
    run_and_write_predictions,
)
from src.gravity_model.gravity_model import GravityModel
from src.simulation.gravity.load_models import discover_age_group_models, load_all_models


def _load_gravity_models() -> dict[str, object]:
    details = get_model_artifact_details("gravity")
    model_paths = discover_age_group_models(Path(details["source_dir"]))
    return load_all_models(model_paths)


def predict_full_results(df: pd.DataFrame, prediction_year: int) -> pd.DataFrame:
    spec = get_model_spec("gravity")
    models = _load_gravity_models()
    gravity_model = GravityModel()

    all_predictions: list[pd.DataFrame] = []
    for age_group in df["age_group"].unique():
        age_group_frame = df.loc[df["age_group"] == age_group].copy()
        gravity_model.set_fitted_model(models[age_group])
        predicted = gravity_model.predict(
            {prediction_year: {age_group: age_group_frame.copy()}},
            str(spec["ground_truth_column"]),
            prediction_year,
            age_group,
        )[prediction_year][age_group].copy()
        predicted["predicted_flow"] = predicted["predicted_flow"].round()
        amount = predicted[str(spec["ground_truth_column"])].copy()
        predicted_values = predicted["predicted_flow"].copy()
        abs_diff = (predicted_values - amount).abs()
        result = pd.DataFrame(
            {
                "area_code_target": predicted["area_code_target"].to_numpy(),
                "area_code_origin": predicted["area_code_origin"].to_numpy(),
                "age_group": [age_group] * len(predicted),
                "Province_of_Origin": predicted["Province_of_Origin"].to_numpy(),
                "target": predicted["target"].to_numpy(),
                "distance_in_meters": predicted["distance_in_meters"].to_numpy(),
                "amount": amount.to_numpy(),
                "predicted": predicted_values.to_numpy(),
                "abs_diff": abs_diff.to_numpy(),
            }
        )
        result["rel_diff"] = result["abs_diff"] / result["amount"]
        all_predictions.append(
            result
        )

    return pd.concat(all_predictions, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate gravity predictions from pretrained artifacts.")
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
    written = run_and_write_predictions(
        model_name="gravity",
        predictor=predict_full_results,
        persist_inputs=not args.skip_persist_inputs,
        overwrite_inputs=args.overwrite_inputs,
    )
    print_written_paths("gravity", written)


if __name__ == "__main__":
    main()
