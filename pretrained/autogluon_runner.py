from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from joblib import parallel_config

from pretrained.common import (
    get_model_artifact_details,
    get_model_spec,
    prepare_model_frame,
    print_written_paths,
    run_and_write_predictions,
)


def predict_full_results(df: pd.DataFrame, prediction_year: int) -> pd.DataFrame:
    details = get_model_artifact_details("autogluon")
    spec = get_model_spec("autogluon")
    prepared = prepare_model_frame(df, relative=bool(spec["relative"]))

    predictor = TabularPredictor.load(str(Path(details["source_dir"]) / "model"))
    with parallel_config(n_jobs=1):
        predictions = predictor.predict(prepared)
    amount = prepared[str(spec["ground_truth_column"])]

    result = pd.DataFrame(
        {
            "area_code_target": prepared["area_code_target"].to_numpy(),
            "area_code_origin": prepared["area_code_origin"].to_numpy(),
            "age_group": prepared["age_group"].to_numpy(),
            "Province_of_Origin": prepared["Province_of_Origin"].to_numpy(),
            "target": prepared["target"].to_numpy(),
            "distance_in_meters": prepared["distance_in_meters"].to_numpy(),
            "amount": amount.to_numpy(),
            "predicted": predictions.to_numpy(),
        }
    )
    result["abs_diff"] = (result["amount"] - result["predicted"]).abs()
    result["rel_diff"] = result["abs_diff"] / result["amount"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate AutoGluon predictions from pretrained artifacts.")
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
        model_name="autogluon",
        predictor=predict_full_results,
        persist_inputs=not args.skip_persist_inputs,
        overwrite_inputs=args.overwrite_inputs,
    )
    print_written_paths("autogluon", written)


if __name__ == "__main__":
    main()
