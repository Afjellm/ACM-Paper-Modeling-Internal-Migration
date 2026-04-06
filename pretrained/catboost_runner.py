from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from pretrained.common import (
    get_model_artifact_details,
    get_model_spec,
    prepare_model_frame,
    print_written_paths,
    run_and_write_predictions,
)


def predict_full_results(df: pd.DataFrame, prediction_year: int) -> pd.DataFrame:
    details = get_model_artifact_details("catboost")
    spec = get_model_spec("catboost")
    prepared = prepare_model_frame(df, relative=bool(spec["relative"]))

    pipeline_dir = Path(details["source_dir"]) / "pipelines"
    all_predictions: list[pd.DataFrame] = []
    for age_group in prepared["age_group"].unique():
        age_group_frame = prepared.loc[prepared["age_group"] == age_group].copy()
        pipeline = load(pipeline_dir / f"best_pipeline_{age_group}.joblib")
        predictions = pd.Series(pipeline.predict(age_group_frame)).clip(lower=0).to_numpy()
        amount = age_group_frame[str(spec["ground_truth_column"])]
        result = pd.DataFrame(
            {
                "area_code_target": age_group_frame["area_code_target"].to_numpy(),
                "area_code_origin": age_group_frame["area_code_origin"].to_numpy(),
                "age_group": age_group_frame["age_group"].to_numpy(),
                "Province_of_Origin": age_group_frame["Province_of_Origin"].to_numpy(),
                "target": age_group_frame["target"].to_numpy(),
                "distance_in_meters": age_group_frame["distance_in_meters"].to_numpy(),
                "amount": amount.to_numpy(),
                "predicted": predictions,
            }
        )
        result["abs_diff"] = (result["amount"] - result["predicted"]).abs()
        result["rel_diff"] = result["abs_diff"] / result["amount"]
        all_predictions.append(result)

    combined = pd.concat(all_predictions, ignore_index=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate CatBoost predictions from pretrained artifacts.")
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
        model_name="catboost",
        predictor=predict_full_results,
        persist_inputs=not args.skip_persist_inputs,
        overwrite_inputs=args.overwrite_inputs,
    )
    print_written_paths("catboost", written)


if __name__ == "__main__":
    main()
