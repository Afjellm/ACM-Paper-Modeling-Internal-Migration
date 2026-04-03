from pathlib import Path
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data_loading.col_normalize import norm_count_cols
from src.data_loading.load_model_input import load_model_input_full
from src.data_loading.rows_per_age_group import get_rows_per_age_group_from_env, rows_per_age_group_label
from src.constrained_xgboost.experiment import run_experiment_and_write_results
from sklearn.metrics import make_scorer, mean_poisson_deviance
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
import numpy as np
import argparse

from src.constrained_xgboost.utils import load_monotone_constraints_from_xlsx

# Training hyperparameters
USE_AGE_GROUPS_AS_SUB_DICT = False
GROUND_TRUTH_COLUMN = "amount_next_year"
NORMALIZE_COUNT_COLS=True
constraint_set = "s1"
EXPERIMENT_NAME = f"xboost_model_{constraint_set}"


# Data loading parameters
START_YEAR = 2018
END_YEAR_EXCLUSIVE = 2024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONSTRAINTS_DIR = Path(__file__).resolve().parent / "constraints"

def create_diff_columns(dataframe, features_for_training) -> List[str]:
    new_features = []

    for col in features_for_training:
        if col == "distance_in_meters":
            continue  # skip distance
        if col.endswith("_origin"):
            base = col.replace("_origin", "")
            target_col = f"{base}_target"
            if target_col in features_for_training:
                diff_col = f"diff_{base}"
                dataframe[diff_col] = dataframe[f"{base}_target"] - dataframe[f"{base}_origin"]
                new_features.append(diff_col)

    return new_features


def clean_xy(X_df: pd.DataFrame, y_ser: pd.Series):
    # keep only numeric columns
    X_num = X_df.select_dtypes(include=[np.number]).copy()

    # coerce any remaining weirdness to numeric (NaNs if bad)
    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    # cast to float32 and force C-contiguous copy
    X_clean = np.ascontiguousarray(X_num.astype(np.float32).to_numpy(copy=True))

    # y: coerce to numeric -> float32 -> 1D contiguous array
    y_clean = pd.to_numeric(y_ser, errors="coerce").astype(np.float32)
    y_clean = np.ascontiguousarray(y_clean.to_numpy(copy=True).ravel())

    return X_clean, y_clean


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_column", type=str, default="amount", help="Ground truth column")
    args = parser.parse_args()
    GROUND_TRUTH_COLUMN = args.ground_truth_column
    rows_per_age_group = get_rows_per_age_group_from_env()

    print(f"using {rows_per_age_group} rows per age group")

    data = load_model_input_full(
        str(DATA_DIR),
        use_age_groups_as_sub_dict=USE_AGE_GROUPS_AS_SUB_DICT,
        rows_per_age_group=rows_per_age_group,
    )

    age_groups = data[2020]["age_group"].unique()

    for year, df in data.items():
        df["within_district"] = (df["area_code_origin"] == df["area_code_target"]).astype(int).astype("category")
        df["is_adjacent"] = df["is_adjacent"].astype("category")
        df["is_covid_year"] = year in [2020, 2021]
        df["is_covid_year"] = df["is_covid_year"].astype("category")
        df["income_ratio"] = df["gross_income_target"] / df["gross_income_origin"] * 100
        df["rent_ratio"] = df["rental_price_target"] / df["rental_price_origin"] * 100
        df["gpreis_ratio"] = df["land_price_target"] / df["land_price_origin"] * 100
        df["gdp_ratio"] = df["gdp_target"] / df["gdp_origin"] * 100
        df["immo_ratio"] = df["real_estate_price_target"] / df["real_estate_price_origin"] * 100
        df = df[df["area_code_origin"]!=df["area_code_target"]]
        data[year]=df

    if NORMALIZE_COUNT_COLS:
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_normalized_cols"
        for year, df in data.items():
            data[year]=norm_count_cols(df)

    EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{GROUND_TRUTH_COLUMN}_{rows_per_age_group_label(rows_per_age_group)}"

    print(f"Training xg boost model using ground truth column {GROUND_TRUTH_COLUMN} in output path {EXPERIMENT_NAME}")

    features_for_training_numeric_all = [
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
        "duration_car_in_min_urban_centre_origin",
        "duration_car_in_min_urban_centre_target",
        "duration_train_in_min_urban_centre_origin",
        "duration_train_in_min_urban_centre_target",
        "rent_ratio",
        "income_ratio",
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
        "is_adjacent",
        "is_covid_year"
    ]

    X_Data = {}
    Y_Data = {}
    Y_AMOUNT_DATA = {}
    all_data_restructured = {}
    for year, df in data.items():
        X_Data[year] = {}
        Y_Data[year] = {}
        Y_AMOUNT_DATA[year] = {}
        all_data_restructured[year] = {}
        for age_group in age_groups:
            df_age = df[df["age_group"] == age_group]

            all_data_restructured[year][age_group] = df_age
            X_Data[year][age_group] = df_age[[col for col in features_for_training_numeric_all if col in df.columns]]
            Y_Data[year][age_group] = df_age[GROUND_TRUTH_COLUMN]
            Y_AMOUNT_DATA[year][age_group] = df_age[GROUND_TRUTH_COLUMN]

    param_grid = {
        "regressor__n_estimators": [300, 500],
        "regressor__learning_rate": [0.1],
        "regressor__max_depth": [5, 10],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.3, 0.6],
        "regressor__colsample_bylevel": [0.6, 0.8]
    }

    poisson_scorer = make_scorer(mean_poisson_deviance, greater_is_better=False)

    fit_years = [[2020, 2021, 2022]]
    for fit_years in fit_years:
        for age_group in age_groups:
            print(f"fitting model for fit years {fit_years} and age_group {age_group}")

            # One hot encode age groups
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", features_for_training_numeric_all)
                ]
            )
            constraints_path = CONSTRAINTS_DIR / f"{constraint_set}.xlsx"
            monotone_constraints = None
            if constraints_path.exists():
                monotone_constraints = load_monotone_constraints_from_xlsx(
                    xlsx_path=constraints_path,
                    age_group_value=age_group,
                    features_for_training_numeric=features_for_training_numeric_all,
                    age_col="age group",
                )
            else:
                print(
                    f"[WARN] Constraint file not found at {constraints_path}. "
                    "Training without monotone constraints."
                )

            xgb_reg = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", XGBRegressor(objective='count:poisson', monotone_constraints=monotone_constraints,
                                           eval_metric='poisson-nloglik'))
            ])

            grid_search = GridSearchCV(
                estimator=xgb_reg,
                param_grid=param_grid,
                cv=5,
                scoring=poisson_scorer,
                verbose=1,
                n_jobs=2
            )

            run_experiment_and_write_results(fit_years, X_Data, Y_Data, all_data_restructured, grid_search,
                                             EXPERIMENT_NAME, START_YEAR,
                                             END_YEAR_EXCLUSIVE, age_group,Y_AMOUNT_DATA, features_for_training_numeric_all)
