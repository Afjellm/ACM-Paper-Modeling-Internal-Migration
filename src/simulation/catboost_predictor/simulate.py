from pathlib import Path
from joblib import load

import pandas as pd
import argparse

from src.data_loading.col_normalize import norm_count_cols
from src.model_artifacts import get_model_output_base


def simulate_data_with_catboost(df, model_name, fit_year, ground_truth_column, area_code, year, full_output_path):
    project_root = Path(__file__).resolve().parents[3]
    base_path = project_root / get_model_output_base()


    age_groups = df["age_group"].unique()
    all_predictions = []
    for age_group in age_groups:
        df_prediction = df[df["age_group"] == age_group]
        path = Path(f"{base_path}/catboost_output/{model_name}/{fit_year}/pipelines/best_pipeline_{age_group}.joblib").resolve()
        pipe = load(path)

        y_simulation = df_prediction[ground_truth_column]
        preds = pipe.predict(df_prediction)

        test_results = pd.DataFrame({
            "area_code_target": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "area_code_target"]),
            "area_code_origin": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "area_code_origin"]),
            "age_group": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "age_group"]),
            "Province_of_Origin": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "Province_of_Origin"]),
            "target": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "target"]),
            "distance_in_meters": df_prediction.index.map(lambda idx: df_prediction.loc[idx, "distance_in_meters"]),
            "amount": y_simulation,
            "predicted": preds,
            "abs_diff": abs(y_simulation - preds),
            "rel_diff": abs(y_simulation - preds) / y_simulation,
        })

        all_predictions.append(test_results)


    test_results = pd.concat(all_predictions)
    test_results["predicted"] = test_results["predicted"].clip(lower=0)
    test_results["year"] = year
    test_results.to_csv(full_output_path)

    grouped = test_results.groupby(
        ["area_code_origin", "area_code_target"], as_index=False
    )[["predicted", "amount"]].sum()

    inward = round(grouped[grouped["area_code_target"] == area_code][["predicted", "amount"]].sum())
    outward = round(grouped[grouped["area_code_origin"] == area_code][["predicted", "amount"]].sum())
    migration_summary = pd.DataFrame({
        "direction": ["inward", "outward"],
        "predicted": [inward["predicted"], outward["predicted"]],
        "amount": [inward["amount"], outward["amount"]]
    })
    return migration_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Catboost simulation in isolated environment")
    parser.add_argument("--df", required=True, help="Input parquet file with features")
    parser.add_argument("--model-name", required=True, help="Path to Catboost model directory")
    parser.add_argument("--fit-year", required=True, help="Path to Catboost model directory")
    parser.add_argument("--ground-truth", required=True, help="Ground truth column name")
    parser.add_argument("--area-code", required=True, help="Area code for context (not used directly in predictions)")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    parser.add_argument("--year", required=True, help="year of simulation")
    parser.add_argument("--detailed-output", required=True, help="year of simulation")
    parser.add_argument("--relative", required=True, help="relative mode")

    args = parser.parse_args()
    relative = args.relative.lower() == "true"

    df = pd.read_csv(args.df)

    if relative:
        df = norm_count_cols(df)

    df["is_adjacent"] = df["is_adjacent"].astype("category")
    df["is_covid_year"] = df["is_covid_year"].astype("category")
    df["income_ratio"] = df["gross_income_target"] / df["gross_income_origin"] * 100
    df["rent_ratio"] = df["rental_price_target"] / df["rental_price_origin"] * 100
    df["gpreis_ratio"] = df["land_price_target"] / df["land_price_origin"] * 100
    df["gdp_ratio"] = df["gdp_target"] / df["gdp_origin"] * 100
    df["immo_ratio"] = df["real_estate_price_target"] / df["real_estate_price_origin"] * 100

    result = simulate_data_with_catboost(
        df=df,
        model_name=args.model_name,
        fit_year=args.fit_year,
        ground_truth_column=args.ground_truth,
        area_code=int(args.area_code),
        year = args.year,
        full_output_path=args.detailed_output
    )
    result.to_csv(args.out, index=False)
