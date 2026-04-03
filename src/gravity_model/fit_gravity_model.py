from typing import Dict, List
from pathlib import Path

import pandas as pd
import pickle

from src.data_loading.load_model_input import load_model_input_full
from src.data_loading.rows_per_age_group import get_rows_per_age_group_from_env, rows_per_age_group_label
from src.gravity_model.gravity_model import GravityModel
import os
import argparse

START_YEAR = 2018
END_YEAR_EXCLUSIVE = 2024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "models" / "output" / "gravity_output"

def evaluate_predictions(data: Dict[int, pd.DataFrame], fit_years: str , gravity_model: GravityModel, output_path: str, ground_truth_column, age_group: str = None) -> pd.DataFrame:
    eval_dict = {}
    save_path = os.path.join(output_path, fit_years)
    os.makedirs(save_path, exist_ok=True)

    offset = 1 if(ground_truth_column == "amount_next_year") else 0

    for key,val in data.items():

        predicted_year = key + offset
        if predicted_year < START_YEAR:
            continue
        if age_group is None:
            pred_data = val
        else:
            pred_data = val[age_group]

        save_path_year = os.path.join(save_path,str(predicted_year), age_group if age_group is not None else "")
        os.makedirs(save_path_year, exist_ok=True)

        pred_data["abs_diff"] = (pred_data["predicted_flow"] - pred_data[ground_truth_column]).abs()
        predicted_df = pred_data[['area_code_origin', 'area_code_target', ground_truth_column, 'predicted_flow', 'abs_diff']]
        predicted_df["year"]=predicted_year
        predicted_df["age_group"] = age_group

        out_migrations = predicted_df.groupby(["area_code_origin"])[
            [ground_truth_column, "predicted_flow"]].sum().reset_index().rename(
            columns={ground_truth_column: "amount_out", "predicted_flow": "predicted_out"}
        )

        in_migrations = predicted_df.groupby(["area_code_target"])[
            [ground_truth_column, "predicted_flow"]].sum().reset_index().rename(
            columns={ground_truth_column: "amount_in", "predicted_flow": "predicted_in"}
        )

        aggregated_migrations = pd.merge(
            in_migrations,
            out_migrations,
            how="outer",
            left_on=["area_code_target"],
            right_on=["area_code_origin"]
        ).fillna(0)

        aggregated_migrations['relative_error'] = (aggregated_migrations['predicted_in'] - aggregated_migrations["amount_in"]).abs() / \
                                       aggregated_migrations["amount_in"]

        eval_dict[predicted_year] = aggregated_migrations

        gravity_model.fitted_model.remove_data()

        with open(os.path.join(save_path_year, "model.pkl"), "wb") as f:
            pickle.dump(gravity_model.fitted_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        gravity_model.fitted_params.to_csv(os.path.join(save_path_year , "fitted_params.csv"))
        gravity_model.fitted_exp_params.to_csv(os.path.join(save_path_year , "fitted_exp_params.csv"))
        gravity_model.standard_errors.to_csv(os.path.join(save_path_year , "standard_errors.csv"))
        gravity_model.pvalues.to_csv(os.path.join(save_path_year , "pvalues.csv"))
        gravity_model.conf_int.to_csv(os.path.join(save_path_year , "conf_int.csv"))

        csv_path = os.path.join(save_path_year, f"evaluation_{predicted_year}.csv")
        full_path = os.path.join(save_path_year, f"full_results.csv")
        # Append the actual grouped_df below the metrics
        aggregated_migrations.to_csv(csv_path, mode='a', index=False)
        predicted_df.to_csv(full_path, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_age_pop", dest="use_age_pop", action="store_true")
    parser.add_argument("--no_use_age_pop", dest="use_age_pop", action="store_false")
    parser.add_argument("--ground_truth_column", type=str, default="amount", help="Ground truth column")

    args = parser.parse_args()

    USE_AGE_GROUPS = True
    GROUND_TRUTH_COLUMN = args.ground_truth_column
    USE_AGE_POP = args.use_age_pop
    rows_per_age_group = get_rows_per_age_group_from_env()

    pop_name = "age_pop" if USE_AGE_POP else "normal_pop"
    ground_truth_name = "prev_year" if GROUND_TRUTH_COLUMN == "amount_next_year" else "current_year"
    EXPERIMENT_NAME = f"gravity_{ground_truth_name}_{pop_name}_{rows_per_age_group_label(rows_per_age_group)}"
    OUTPUT_PATH = OUTPUT_ROOT / EXPERIMENT_NAME


    data = load_model_input_full(
        str(DATA_DIR),
        use_age_groups_as_sub_dict=USE_AGE_GROUPS,
        rows_per_age_group=rows_per_age_group,
    )

    print(f"Running gravity model {EXPERIMENT_NAME}")

    for year, agegroup_data in data.items():
        for age_group, df in agegroup_data.items():
            df["income_ratio"] = df["gross_income_target"] / df["gross_income_origin"] * 100
            df["rent_ratio"] = df["rental_price_target"] / df["rental_price_origin"] * 100
            df["gpreis_ratio"] = df["land_price_target"] / df["land_price_origin"] * 100
            df["gdp_ratio"] = df["gdp_target"] / df["gdp_origin"] * 100
            df["immo_ratio"] = df["real_estate_price_target"] / df["real_estate_price_origin"] * 100
            df = df[df["area_code_origin"] != df["area_code_target"]]
            agegroup_data[age_group] = df

    prediction_start_year = START_YEAR - 1 if GROUND_TRUTH_COLUMN == "amount_next_year" else START_YEAR

    gravity_model = GravityModel()
    fit_years = [[2019,2020,2021,2022]]
    age_groups = data[2018].keys()
    for fit_year_range in fit_years:
        for age_group in age_groups:

            fit_year_str = "_".join(str(year) for year in fit_year_range)

            data_to_fit = pd.concat([data[year][age_group] for year in fit_year_range], axis=0, ignore_index=True)

            gravity_model.fit_model(data_to_fit,GROUND_TRUTH_COLUMN, age_group, USE_AGE_POP)
            predictions = gravity_model.predict(data,GROUND_TRUTH_COLUMN, prediction_start_year, age_group)
            evaluate_predictions(predictions, fit_year_str, gravity_model, OUTPUT_PATH, GROUND_TRUTH_COLUMN, age_group)
