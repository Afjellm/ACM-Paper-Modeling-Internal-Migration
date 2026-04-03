import os
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "models" / "output" / "catboost_output"


def run_experiment_and_write_results(fit_years, X_DATA, Y_DATA, data, grid_search, experiment_name, START_YEAR, END_YEAR, age_group, Y_AMOUNT_DATA):

    # Train-test split
    fit_year_str = "_".join(str(year) for year in fit_years)

    X_TRAIN = pd.concat([X_DATA[year][age_group] for year in fit_years], axis=0, ignore_index=True)
    Y_TRAIN = pd.concat([Y_DATA[year][age_group] for year in fit_years], axis=0, ignore_index=True)
    Y_AMOUNT_TRAIN = pd.concat([Y_AMOUNT_DATA[year][age_group] for year in fit_years], axis=0, ignore_index=True)
    X_train, X_test, y_train, y_test, y_amount_train, y_amount_test = train_test_split(X_TRAIN, Y_TRAIN,Y_AMOUNT_TRAIN, test_size=0.1, random_state=100)

    grid_search.fit(X_train, y_train)
    regressor = grid_search.best_estimator_
    best_pipeline = grid_search.best_estimator_

    regresser_trees = regressor.named_steps["regressor"]
    importances = regresser_trees.feature_importances_


    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': X_DATA[fit_years[0]][age_group].columns,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    fit_year_save_path = OUTPUT_PATH / experiment_name / fit_year_str
    os.makedirs(fit_year_save_path, exist_ok=True)
    pipeline_save_path = fit_year_save_path / "pipelines"
    os.makedirs(pipeline_save_path, exist_ok=True)
    dump(best_pipeline, pipeline_save_path / f"best_pipeline_{age_group}.joblib")

    with open(pipeline_save_path / "features_importances.csv", "w") as f:
        feature_importance_df.to_csv(f, index=False)

    for experiment_year in range(START_YEAR, END_YEAR):
        X_TEST = X_DATA[experiment_year][age_group]
        Y_AMOUNT_TEST = Y_AMOUNT_DATA[experiment_year][age_group]

        y_test_pred = regressor.predict(X_TEST)

        # Create a DataFrame with actual vs predicted values for the test set
        test_results = pd.DataFrame({
            "area_code_target": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "area_code_target"]),
            "area_code_origin": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "area_code_origin"]),
            "age_group": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "age_group"]),
            "Province_of_Origin": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "Province_of_Origin"]),
            "target": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "target"]),
            "distance_in_meters": X_TEST.index.map(lambda idx: data[experiment_year][age_group].loc[idx, "distance_in_meters"]),
            "amount": Y_AMOUNT_TEST,
            "predicted": y_test_pred,
            "abs_diff": abs(Y_AMOUNT_TEST - y_test_pred),
            "rel_diff": abs(Y_AMOUNT_TEST - y_test_pred) / Y_AMOUNT_TEST,
        })



        aggregated_migrations_with_within_migration = create_aggregated_migrations_from_predicted_dataframe(
            test_results)

        without_within_migration = test_results[test_results["area_code_origin"] != test_results["area_code_target"]]

        aggregated_migrations_without_within_migration = create_aggregated_migrations_from_predicted_dataframe(
            without_within_migration)

        # Write results
        fit_year_path = OUTPUT_PATH / experiment_name / str(fit_year_str)
        experiment_year_save_path = fit_year_path / str(experiment_year)
        os.makedirs(experiment_year_save_path, exist_ok=True)

        age_group_output_path = experiment_year_save_path / str(age_group)
        os.makedirs(age_group_output_path, exist_ok=True)

        with open(age_group_output_path / "results_with_within_migration.csv", "w") as f:
            aggregated_migrations_with_within_migration.to_csv(f, index=False)

        with open(age_group_output_path / "full_results.csv", "w") as f:
            test_results.to_csv(f, index=False)

        with open(age_group_output_path / "results_without_within_migration.csv", "w") as f:
            aggregated_migrations_without_within_migration.to_csv(f, index=False)


def create_aggregated_migrations_from_predicted_dataframe(df):
    out_migrations = df.groupby(["area_code_origin"])[
        ["amount", "predicted"]].sum().reset_index().rename(
        columns={"amount": "amount_out", "predicted": "predicted_out"}
    )

    # Incoming migrations grouped by target and age — FIXED: groupby([...]) instead of groupby(..., ...)
    in_migrations = df.groupby(["area_code_target"])[
        ["amount", "predicted"]].sum().reset_index().rename(
        columns={"amount": "amount_in", "predicted": "predicted_in"}
    )
    # Aggregate migrations **arriving in each target group**
    aggregated_migrations = pd.merge(
        in_migrations,
        out_migrations,
        how="outer",
        left_on=["area_code_target"],
        right_on=["area_code_origin"]
    ).fillna(0)

    # Calculate relative errors
    aggregated_migrations["Relative Error Out"] = aggregated_migrations.apply(
        lambda row: abs(row["amount_out"] - row["predicted_out"]) / row["amount_out"]
        if row["amount_out"] != 0 else 0,
        axis=1
    )

    aggregated_migrations["Relative Error In"] = aggregated_migrations.apply(
        lambda row: abs(row["amount_in"] - row["predicted_in"]) / row["amount_in"]
        if row["amount_in"] != 0 else 0,
        axis=1
    )

    return aggregated_migrations
