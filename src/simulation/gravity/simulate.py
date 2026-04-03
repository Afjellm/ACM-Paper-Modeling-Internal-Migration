from src.gravity_model.gravity_model import GravityModel

from typing import Hashable
import pandas as pd


def simulate_data_with_gravity(df, gravity_models, ground_truth_column, area_code, year):

    age_groups = df['age_group'].unique()
    data_to_simulate = {}
    for age_group in age_groups:
        data_to_simulate[age_group] = df[df['age_group'] == age_group].copy()

    return in_out_flows(data_to_simulate, gravity_models, area_code, year, ground_truth_column)



def in_out_flows(data_to_simulate, models, area_code: Hashable, year: int, ground_truth_column: str) -> pd.DataFrame:
    """
    Return (amount_in, amount_out, predicted_in, predicted_out) for a given area_code.

    Expects columns:
      - area_code_origin, area_code_target
      - amount (ground truth), predicted_flow
    """

    gravity_model = GravityModel()
    age_groups = data_to_simulate.keys()

    for age_group in age_groups:
        gravity_model.set_fitted_model(models[age_group])
        predictions = gravity_model.predict({year: data_to_simulate}, ground_truth_column, year,
                                            age_group)
        predictions[year][age_group]["predicted_flow"] = round(
        predictions[year][age_group]["predicted_flow"])

    df = pd.concat(predictions[year].values(), ignore_index=True)

    out_amount = df.groupby("area_code_origin")[ground_truth_column].sum(min_count=1)
    out_pred = df.groupby("area_code_origin")["predicted_flow"].sum(min_count=1)

    in_amount = df.groupby("area_code_target")[ground_truth_column].sum(min_count=1)
    in_pred = df.groupby("area_code_target")["predicted_flow"].sum(min_count=1)

    # Safely pull values (0.0 if missing/NaN)
    amount_out = float(out_amount.get(area_code, 0.0) or 0.0)
    predicted_out = float(out_pred.get(area_code, 0.0) or 0.0)
    amount_in = float(in_amount.get(area_code, 0.0) or 0.0)
    predicted_in = float(in_pred.get(area_code, 0.0) or 0.0)

    # Return the same shape as your `migration_summary`
    migration_summary = pd.DataFrame({
        "direction": ["inward", "outward"],
        "predicted": [predicted_in, predicted_out],
        "amount": [amount_in, amount_out],
    })

    return migration_summary

