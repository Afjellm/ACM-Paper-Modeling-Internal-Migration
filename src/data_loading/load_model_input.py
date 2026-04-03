from typing import Dict, List
import os
import re

import pandas as pd


FULL_COLUMNS = [
    "area_code_origin",
    "age_group",
    "area_code_target",
    "amount",
    "amount_prev_year",
    "amount_next_year",
    "Province_of_Origin",
    "target",
    "bezirk_origin",
    "time_origin",
    "population_total_origin",
    "unemp_origin",
    "outside_labour_force_origin",
    "unemp_rate_origin",
    "rental_price_origin",
    "gross_income_origin",
    "download_speed_origin",
    "flood_risk_origin",
    "nuts3_origin",
    "gdp_origin",
    "schools_origin",
    "area_in_km2_origin",
    "pop_density_origin",
    "real_estate_price_origin",
    "duration_car_in_min_urban_centre_origin",
    "duration_train_in_min_urban_centre_origin",
    "c_500_origin",
    "c_1000_origin",
    "c_10_19_origin",
    "c_20_49_origin",
    "c_50_99_origin",
    "c_250_499_origin",
    "c_100_250_origin",
    "rural_level_origin",
    "district_type_origin",
    "land_price_origin",
    "universities_within_5km_origin",
    "fhs_within_5km_origin",
    "year_origin",
    "rental_rate_origin",
    "w_permit_1_2_origin",
    "w_permit_3_10_origin",
    "w_permit_11_origin",
    "green_ratio_origin",
    "green_with_arable_ratio_origin",
    "bezirk_target",
    "time_target",
    "population_total_target",
    "unemp_target",
    "outside_labour_force_target",
    "unemp_rate_target",
    "rental_price_target",
    "gross_income_target",
    "download_speed_target",
    "flood_risk_target",
    "nuts3_target",
    "gdp_target",
    "schools_target",
    "area_in_km2_target",
    "pop_density_target",
    "real_estate_price_target",
    "duration_car_in_min_urban_centre_target",
    "duration_train_in_min_urban_centre_target",
    "c_500_target",
    "c_1000_target",
    "c_10_19_target",
    "c_20_49_target",
    "c_50_99_target",
    "c_250_499_target",
    "c_100_250_target",
    "rural_level_target",
    "district_type_target",
    "land_price_target",
    "universities_within_5km_target",
    "fhs_within_5km_target",
    "year_target",
    "rental_rate_target",
    "w_permit_1_2_target",
    "w_permit_3_10_target",
    "w_permit_11_target",
    "green_ratio_target",
    "green_with_arable_ratio_target",
    "direction",
    "population_within_age_group_origin",
    "population_within_age_group_target",
    "is_adjacent",
    "distance_in_meters"
]


def load_model_input_metadata(metadata_output_base_path: str, years: List[int] = None) -> Dict[int, pd.DataFrame]:
    metadata_by_year = {}
    for item in os.listdir(metadata_output_base_path):
        full_path = os.path.join(metadata_output_base_path, item)
        if os.path.isdir(full_path) and re.fullmatch(r"\d{4}", item):
            year = int(item)
            csv_path = os.path.join(full_path, "metadata.csv")
            if os.path.exists(csv_path) and (years is None or year in years):
                metadata_by_year[year] = pd.read_csv(csv_path)
    return metadata_by_year


def _read_year_file(base_path: str, year: int, filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(base_path, str(year), filename))


def _metadata_with_suffix(metadata: pd.DataFrame, suffix: str) -> pd.DataFrame:
    metadata = metadata.rename(columns={"area_code": f"area_code_{suffix}"})
    rename_map = {col: f"{col}_{suffix}" for col in metadata.columns if col != f"area_code_{suffix}"}
    return metadata.rename(columns=rename_map)


def _build_population_by_age(migration: pd.DataFrame) -> pd.DataFrame:
    return (
        migration[["area_code_origin", "age_group", "population_within_age_group_origin"]]
        .drop_duplicates()
        .rename(
            columns={
                "area_code_origin": "area_code_target",
                "population_within_age_group_origin": "population_within_age_group_target",
            }
        )
    )


def _build_full_frame(base_path: str, year: int) -> pd.DataFrame:
    migration = _read_year_file(base_path, year, "migration.csv").copy()
    metadata = _read_year_file(base_path, year, "metadata.csv")
    distance_matrix = (
        _read_year_file(base_path, year, "distance_matrix.csv")
        .drop_duplicates(subset=["area_code_origin", "area_code_target"])
    )

    population_target = _build_population_by_age(migration)

    if "amount_next_year" not in migration.columns:
        migration["amount_next_year"] = migration["amount"]

    migration["Province_of_Origin"] = migration["bezirk_origin"]
    migration["target"] = migration["bezirk_target"]
    migration = migration.drop(columns=["bezirk_origin", "bezirk_target"])

    origin_metadata = _metadata_with_suffix(metadata, "origin")
    target_metadata = _metadata_with_suffix(metadata, "target")

    full = migration.merge(origin_metadata, on="area_code_origin", how="left")
    full = full.merge(target_metadata, on="area_code_target", how="left")
    full = full.merge(population_target, on=["area_code_target", "age_group"], how="left")
    full = full.merge(distance_matrix, on=["area_code_origin", "area_code_target"], how="left")

    full.loc[full["area_code_origin"] == full["area_code_target"], "distance_in_meters"] = pd.NA
    full.loc[full["area_code_origin"] == full["area_code_target"], "is_adjacent"] = 0

    if "is_adjacent" not in full.columns:
        full["is_adjacent"] = 0

    return full[FULL_COLUMNS]


def load_model_input_full(
    metadata_output_base_path: str,
    years: List[int] = None,
    use_age_groups_as_sub_dict: bool = False,
    rows_per_age_group: int = None,
) -> Dict[int, pd.DataFrame]:
    metadata_by_year = {}
    for item in os.listdir(metadata_output_base_path):
        full_path = os.path.join(metadata_output_base_path, item)
        if not (os.path.isdir(full_path) and re.fullmatch(r"\d{4}", item)):
            continue

        year = int(item)
        if years is not None and year not in years:
            continue

        required_files = ["migration.csv", "metadata.csv", "distance_matrix.csv"]
        if not all(os.path.exists(os.path.join(full_path, filename)) for filename in required_files):
            continue

        data = _build_full_frame(metadata_output_base_path, year)
        if not use_age_groups_as_sub_dict:
            metadata_by_year[year] = data
        else:
            age_groups = data["age_group"].unique()
            metadata_by_year[year] = {
                age_group: data[data["age_group"] == age_group].copy()
                for age_group in age_groups
            }

    if rows_per_age_group is not None:
        for year, data in metadata_by_year.items():
            if use_age_groups_as_sub_dict:
                for age_group, df in data.items():
                    metadata_by_year[year][age_group] = df.head(rows_per_age_group)
            else:
                metadata_by_year[year] = data.groupby("age_group", group_keys=False).head(rows_per_age_group)

    return metadata_by_year
