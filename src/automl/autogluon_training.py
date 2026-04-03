import argparse
import os
from pathlib import Path

from autogluon.tabular import TabularPredictor

from src.automl.cpc_score_metric import ag_cpc_scorer
from src.automl.experiment import run_experiment_and_write_results
from src.data_loading.col_normalize import norm_count_cols
from src.data_loading.load_model_input import load_model_input_full
from src.data_loading.rows_per_age_group import (
    get_rows_per_age_group_from_env,
    parse_positive_int,
    rows_per_age_group_label,
)

# Training hyperparameters
GROUND_TRUTH_COLUMN = "amount"
START_YEAR = 2018
END_YEAR_EXCLUSIVE = 2024
EXPERIMENT_NAME = "autogluon"
NORMALIZE_COUNT_COLS = False


# Data loading parameters
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "models" / "output" / "autogluon_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_column", type=str, default="amount", help="Ground truth column")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ground_truth_column = args.ground_truth_column
    rows_per_age_group = get_rows_per_age_group_from_env()
    time_limit = parse_positive_int(os.getenv("TIME_LIMIT"), default=14400, name="TIME_LIMIT")

    data = load_model_input_full(
        str(DATA_DIR),
        use_age_groups_as_sub_dict=False,
        rows_per_age_group=rows_per_age_group,
    )

    for year, df in data.items():
        df["is_adjacent"] = df["is_adjacent"].astype("category")
        df["is_covid_year"] = year in [2020, 2021]
        df["is_covid_year"] = df["is_covid_year"].astype("category")
        df["income_ratio"] = df["gross_income_target"] / df["gross_income_origin"] * 100
        df["rent_ratio"] = df["rental_price_target"] / df["rental_price_origin"] * 100
        df["gpreis_ratio"] = df["land_price_target"] / df["land_price_origin"] * 100
        df["gdp_ratio"] = df["gdp_target"] / df["gdp_origin"] * 100
        df["immo_ratio"] = df["real_estate_price_target"] / df["real_estate_price_origin"] * 100
        df = df[df["area_code_origin"] != df["area_code_target"]]
        data[year] = df

    experiment_name = EXPERIMENT_NAME
    if NORMALIZE_COUNT_COLS:
        experiment_name = f"{experiment_name}_normalized_cols"
        for year, df in data.items():
            data[year] = norm_count_cols(df)

    experiment_name = f"{experiment_name}_{ground_truth_column}_{rows_per_age_group_label(rows_per_age_group)}"
    print(f"Training AutoML model using ground truth column {ground_truth_column} in output path {experiment_name}")
    features_for_training_numeric = [
        "population_within_age_group_origin",
        "population_within_age_group_target",
        "population_total_origin",
        "population_total_target",
        "gdp_ratio",
        "schools_origin",
        "schools_target",
        "universities_within_5km_origin",
        "universities_within_5km_target",
        "distance_in_meters",
        "income_ratio",
        "duration_car_in_min_urban_centre_origin",
        "duration_car_in_min_urban_centre_target",
        "duration_train_in_min_urban_centre_origin",
        "duration_train_in_min_urban_centre_target",
        "rent_ratio",
        "gpreis_ratio",
        "c_500_origin",
        "c_500_target",
        "c_1000_origin",
        "c_1000_target",
        "c_50_99_origin",
        "c_50_99_target",
        "c_10_19_origin",
        "c_10_19_target",
        "c_20_49_origin",
        "c_20_49_target",
        "c_250_499_origin",
        "c_250_499_target",
        "c_100_250_origin",
        "c_100_250_target",
        "w_permit_11_target",
        "w_permit_3_10_target",
        "w_permit_1_2_target",
        "green_ratio_origin",
        "green_ratio_target",
        "immo_ratio",
        "rental_rate_origin",
        "rental_rate_target",
        "flood_risk_origin",
        "flood_risk_target",
        "download_speed_origin",
        "download_speed_target",
        ground_truth_column,
    ]

    categorical = ["age_group", "rural_level_origin", "rural_level_target", "is_covid_year", "is_adjacent"]
    features_to_train_on = features_for_training_numeric + categorical

    print(f"Start training {experiment_name} using cpc score")

    x_data = {}
    y_data = {}
    y_amount_data = {}
    for year, df in data.items():
        for col in categorical:
            df[col] = df[col].astype("category")
        x_data[year] = df[[col for col in features_to_train_on if col in df.columns]]
        y_data[year] = df[ground_truth_column]
        y_amount_data[year] = df[ground_truth_column]

    fit_years = [[2019, 2020, 2021, 2022]]
    for fit_year in fit_years:
        fit_year_str = "_".join(str(year) for year in fit_year)
        model_save_path = str(OUTPUT_PATH / experiment_name / fit_year_str / "model")
        os.makedirs(model_save_path, exist_ok=True)

        predictor = TabularPredictor(
            label=ground_truth_column,
            problem_type="regression",
            eval_metric=ag_cpc_scorer,
            path=model_save_path,
        )

        run_experiment_and_write_results(
            fit_year,
            x_data,
            y_data,
            data,
            predictor,
            experiment_name,
            START_YEAR,
            END_YEAR_EXCLUSIVE,
            ground_truth_column,
            str(OUTPUT_PATH),
            y_amount_data,
            time_limit,
        )
