from __future__ import annotations

import os

from src.data_loading.rows_per_age_group import (
    get_rows_per_age_group_from_env,
    rows_per_age_group_label,
)


def get_model_output_base() -> str:
    return os.getenv("MODEL_OUTPUT_BASE", "models/output").rstrip("/\\")


def build_model_artifact_config() -> dict[str, dict[str, str]]:
    rows_label = rows_per_age_group_label(get_rows_per_age_group_from_env())
    output_base = get_model_output_base()
    return {
        "xgboost": {
            "folder": f"{output_base}/xgboost_output/xboost_model_s1_normalized_cols_amount_next_year_{rows_label}",
            "fit_year": "2020_2021_2022",
            "model_name": f"xboost_model_s1_normalized_cols_amount_next_year_{rows_label}",
        },
        "constrained_catboost": {
            "folder": f"{output_base}/catboost_output/caboost_s1_normalized_cols_amount_{rows_label}",
            "fit_year": "2020_2021_2022_2023",
            "model_name": f"caboost_s1_normalized_cols_amount_{rows_label}",
        },
        "autogluon": {
            "folder": f"{output_base}/autogluon_output/autogluon_amount_{rows_label}",
            "fit_year": "2019_2020_2021_2022",
            "model_name": f"autogluon_amount_{rows_label}",
        },
        "gravity": {
            "folder": f"{output_base}/gravity_output/gravity_prev_year_age_pop_{rows_label}",
            "fit_year": "2019_2020_2021_2022",
            "model_name": f"gravity_prev_year_age_pop_{rows_label}",
        },
    }
