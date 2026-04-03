import os

import json
import pandas as pd
from autogluon.tabular import TabularPredictor

NUM_BAG_FOLD = 5
NUM_STACK_LEVELS = 2

def run_experiment_and_write_results(fit_years, X_Data, Y_Data, data, predictor: TabularPredictor, experiment_name, START_YEAR, END_YEAR, GROUND_TRUTH_COLUMN, OUTPUT_PATH, Y_AMOUNT_DATA, time_limit: int):

    fit_year_str = "_".join(str(year) for year in fit_years)
    save_path = OUTPUT_PATH + "/" + experiment_name + "/" + fit_year_str
    os.makedirs(save_path, exist_ok=True)

    save_path_hyperparam = OUTPUT_PATH + "/" + experiment_name + "/" + "hyperparams.json"

    hyperparams = {
        "TIME_LIMIT": time_limit,
        "NUM_BAG_FOLD": NUM_BAG_FOLD,
        "NUM_STACK_LEVELS": NUM_STACK_LEVELS
    }

    if not os.path.exists(save_path_hyperparam):
        os.makedirs(os.path.dirname(save_path_hyperparam), exist_ok=True)
        with open(save_path_hyperparam, "w") as f:
            json.dump(hyperparams, f, indent=4)

    X_TRAIN = pd.concat([X_Data[year] for year in fit_years], axis=0, ignore_index=True)

    predictor.fit(
        train_data=X_TRAIN,
        presets="best_quality",          # try "medium_quality" for speed
        time_limit=time_limit,                 # seconds; set to what you can afford
        num_bag_folds=NUM_BAG_FOLD,                 # ensembling via bagging
        num_stack_levels=NUM_STACK_LEVELS,              # stacking
        # hyperparameters="default"      # or provide dict to include/exclude models
    )
    leaderboard_df = predictor.leaderboard(silent=True)
    best_model = leaderboard_df.loc[0, "model"]
    # Compute importances on a validation set (recommended)
    fi = predictor.feature_importance(
        data=X_TRAIN,
        model=best_model  # use specific model name
    )

    # Make it look like your current dataframe
    feature_importance_df = fi.reset_index().rename(
        columns={"index": "Feature", "importance": "Importance"}
    ).sort_values("Importance", ascending=False)


    with open(os.path.join(save_path, "features_importances.csv"), "w") as f:
        feature_importance_df.to_csv(f, index=False)


    for experiment_year in range(START_YEAR, END_YEAR):
        X_TEST = X_Data[experiment_year].copy()
        Y_TEST = Y_AMOUNT_DATA[experiment_year]

        X_TEST.drop(columns=[GROUND_TRUTH_COLUMN], inplace=True)
        y_test_pred = predictor.predict(X_TEST)

        # Create a DataFrame with actual vs predicted values for the test set
        test_results = pd.DataFrame({
            "area_code_target": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "area_code_target"]),
            "area_code_origin": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "area_code_origin"]),
            "age_group": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "age_group"]),
            "Province_of_Origin": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "Province_of_Origin"]),
            "target": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "target"]),
            "distance_in_meters": X_TEST.index.map(lambda idx: data[experiment_year].loc[idx, "distance_in_meters"]),
            "amount": Y_TEST,
            "predicted": y_test_pred,
            "abs_diff": abs(Y_TEST - y_test_pred),
            "rel_diff": abs(Y_TEST - y_test_pred) / Y_TEST,
        })
        # Aggregate migrations **leaving each origin group** (only cross-group)

        aggregated_migrations_with_within_migration = create_aggregated_migrations_from_predicted_dataframe(test_results)

        without_within_migration = test_results[test_results["area_code_origin"]!=test_results["area_code_target"]]

        aggregated_migrations_without_within_migration = create_aggregated_migrations_from_predicted_dataframe(without_within_migration)

        # Write results
        fit_year_path = OUTPUT_PATH + "/" + experiment_name + "/" + str(fit_year_str)
        save_path = fit_year_path + "/" + str(experiment_year)
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "results_with_within_migration.csv"), "w") as f:
            aggregated_migrations_with_within_migration.to_csv(f, index=False)

        with open(os.path.join(save_path, "results_without_within_migration.csv"), "w") as f:
            aggregated_migrations_without_within_migration.to_csv(f, index=False)

        with open(os.path.join(save_path, "full_results.csv"), "w") as f:
            test_results.to_csv(f, index=False)

        predictor.save(fit_year_path)

        age_groups = X_TRAIN['age_group'].unique()
        for age_group in age_groups:
            age_group_output_path = save_path + "/" + age_group
            os.makedirs(age_group_output_path, exist_ok=True)

            age_group_results = test_results[test_results["age_group"] == age_group]

            aggregated_migrations_age_group_with_within_migration = create_aggregated_migrations_from_predicted_dataframe(age_group_results)

            age_group_without_within_migration = age_group_results[age_group_results["area_code_origin"] != age_group_results["area_code_target"]]

            aggregated_migrations_age_group_without_within_migration = create_aggregated_migrations_from_predicted_dataframe(
                age_group_without_within_migration)

            with open(os.path.join(age_group_output_path, "results_with_within_migration.csv"), "w") as f:
                aggregated_migrations_age_group_with_within_migration.to_csv(f, index=False)

            with open(os.path.join(age_group_output_path, "results_without_within_migration.csv"), "w") as f:
                aggregated_migrations_age_group_without_within_migration.to_csv(f, index=False)


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
