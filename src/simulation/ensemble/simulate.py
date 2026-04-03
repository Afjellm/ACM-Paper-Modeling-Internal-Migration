import os
from pathlib import Path
import pandas as pd
import argparse

from autogluon.tabular import TabularPredictor

from src.data_loading.col_normalize import norm_count_cols

os.environ["AUTOGluon_Allow_Model_Move"] = "True"

def simulate_data_with_ensemble(df, model_name, fit_year, ground_truth_column, area_code):
    project_root = Path(__file__).resolve().parents[3]
    base_path = project_root / "models" / "output" / "autogluon_output"

    path = Path(f"{base_path}/{model_name}/{fit_year}/model").resolve()
    predictor = TabularPredictor.load(str(path))

    y_simulation = df[ground_truth_column]
    predictions = predictor.predict(df)


    test_results = pd.DataFrame({
        "area_code_target": df.index.map(lambda idx: df.loc[idx, "area_code_target"]),
        "area_code_origin": df.index.map(lambda idx: df.loc[idx, "area_code_origin"]),
        "age_group": df.index.map(lambda idx: df.loc[idx, "age_group"]),
        "Province_of_Origin": df.index.map(lambda idx: df.loc[idx, "Province_of_Origin"]),
        "target": df.index.map(lambda idx: df.loc[idx, "target"]),
        "distance_in_meters": df.index.map(lambda idx: df.loc[idx, "distance_in_meters"]),
        "amount": y_simulation,
        "predicted": predictions,
        "abs_diff": abs(y_simulation - predictions),
        "rel_diff": abs(y_simulation - predictions) / y_simulation,
    })

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
    parser = argparse.ArgumentParser(description="Run AutoGluon simulation in isolated environment")
    parser.add_argument("--df", required=True, help="Input parquet file with features")
    parser.add_argument("--model-name", required=True, help="Path to AutoGluon model directory")
    parser.add_argument("--fit-year", required=True, help="Path to AutoGluon model directory")
    parser.add_argument("--ground-truth", required=True, help="Ground truth column name")
    parser.add_argument("--area-code", required=True, help="Area code for context (not used directly in predictions)")
    parser.add_argument("--out", required=True, help="Output parquet file path")
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

    result = simulate_data_with_ensemble(
        df=df,
        model_name=args.model_name,
        fit_year=args.fit_year,
        ground_truth_column=args.ground_truth,
        area_code=int(args.area_code),
    )
    result.to_csv(args.out, index=False)