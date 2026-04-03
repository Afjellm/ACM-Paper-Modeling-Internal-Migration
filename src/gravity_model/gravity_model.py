from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class GravityModel():
    def __init__(self):
        super().__init__()
        self.fitted_model = None
        self.best_vars = None
        self.fitted_params = None
        self.fitted_exp_params = None
        self.standard_errors = None
        self.pvalues = None
        self.conf_int = None
        self.log_data = True

    def set_fitted_model(self, model):
        self.fitted_model = model

    def preprocess_data(self, df: pd.DataFrame, ground_truth_colum:str, age_group: str = None) -> pd.DataFrame:

        df['log_pop_origin'] = np.log(df['population_total_target'])
        df['log_pop_dest'] = np.log(df['population_total_origin'])
        df['log_population_within_age_group_origin'] = np.log(df['population_within_age_group_origin'])
        df['log_population_within_age_group_target'] = np.log(df['population_within_age_group_target'])
        df['log_distance'] = np.log(df['distance_in_meters'])

        if self.log_data:
            cols_to_log = [
                "c_10_19",
                "c_20_49",
                "c_50_99",
                "c_250_499",
                "c_100_250",
                'gross_income',
                'schools',
                'real_estate_price',
                'duration_car_in_min_urban_centre',
                'duration_train_in_min_urban_centre',
                'c_500',
                'universities_within_5km',
                'fhs_within_5km',
                'c_1000',
                'w_permit_1_2',
                'w_permit_3_10',
                'w_permit_11',
                'rental_price',
                'land_price',
                'download_speed',
                'gdp'
            ]

            # Log-transform all other numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                if col.replace("_origin","") in cols_to_log or col.replace("_target","") in cols_to_log:
                    log_col = f'log_{col}'
                    df[log_col] = np.log1p(df[col])


        if age_group is not None:
            df_filtered = df[df['area_code_origin'] != df['area_code_target']]
            return df_filtered

        group_cols = ['area_code_origin', 'area_code_target']
        agg_dict = {col: 'first' for col in df.columns if col not in group_cols + [ground_truth_colum, 'age_group']}
        agg_dict[ground_truth_colum] = 'sum'

        df_grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
        df_filtered = df_grouped[df_grouped['area_code_origin'] != df_grouped['area_code_target']]

        return df_filtered

    def fit_model(self, df: pd.DataFrame, ground_truth_colum:str, age_group: str = None, use_age_pop: bool = False):
        data_to_fit = self.preprocess_data(df, ground_truth_colum, age_group)
        postfixes = ["target", "origin"]
        base_vars = ["green_ratio", "unemp", "rental_rate","flood_risk",]
        diadic_vars = ["income_ratio", "gdp_ratio", "gdp_ratio", "rent_ratio"]

        vars_with_potential_log = ["duration_car_in_min_urban_centre", "schools", "c_500", "c_1000","c_50_99", "c_250_499", "c_100_250", "c_10_19",
                                   "c_20_49", "download_speed", "land_price", "universities_within_5km", "fhs_within_5km", "w_permit_1_2", "w_permit_3_10", "w_permit_11"]

        transformed_base_vars = [f"{var}_{pfx}" for var in base_vars for pfx in postfixes]
        if self.log_data:
            all_vars = [*[f"log_{var}_{pfx}" for var in vars_with_potential_log for pfx in postfixes], *transformed_base_vars]
        else:
            all_vars = [*[f"{var}_{pfx}" for var in vars_with_potential_log for pfx in postfixes], *transformed_base_vars]

        all_vars = [*all_vars, *diadic_vars]

        def stepwise_selection(data, response, candidates,
                               initial_features=None,
                               direction='forward'):
            print("start stepwise selection")
            import warnings
            warnings.filterwarnings("ignore")
            model = None
            included = list(initial_features) if initial_features else []
            best_aic = float('inf')
            while True:
                changed = False
                candidates_set = list(set(candidates) - set(included))
                if direction == 'forward':
                    to_test = included + [c for c in candidates_set]
                else:  # backward or mixed
                    to_test = [f for f in included] + [f for f in candidates_set]
                aic_list = []
                for feature in candidates_set:

                    if not use_age_pop:
                        fixed_features = "log_pop_origin + log_pop_dest + log_distance"
                    else:
                        fixed_features = "log_population_within_age_group_origin + log_population_within_age_group_target + log_distance"

                    try:
                        test_features = included + [feature]
                        formula = f"{response} ~ {fixed_features} + " + " + ".join(test_features)
                        model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()
                        aic_list.append((model.aic, feature))
                    except Exception as e:
                        print(f"Error occured with {e}")
                        continue
                if not aic_list:
                    break
                aic_list.sort()
                best_new_aic, best_feature = aic_list[0]
                if best_new_aic < best_aic:
                    included.append(best_feature)
                    best_aic = best_new_aic
                    changed = True
                if not changed:
                    break
            return included, best_aic, model

        best_vars, best_aic, model = stepwise_selection(data_to_fit, ground_truth_colum, all_vars)
        print(f"Best AIC: {best_aic}")
        print(f"Best variables: {best_vars}")

        self.best_vars = best_vars
        self.fitted_params = model.params
        self.fitted_exp_params = model.params.apply(np.exp)
        self.standard_errors = model.bse
        self.pvalues = model.pvalues
        self.conf_int = model.conf_int()
        self.fitted_model = model

    def predict(self, data: Dict[int, pd.DataFrame],ground_truth_colum:str, prediction_start_year:int, age_group: str = None) -> Dict[int, pd.DataFrame]:
        if self.fitted_model is None:
            raise SystemError("Model not fitted yet. First call fit_model")

        for key, df in data.items():
            if key < prediction_start_year:
                continue
            if age_group is None:
                data[key] = self.preprocess_data(data[key], ground_truth_colum)
                data[key]['predicted_flow'] = self.fitted_model.predict(data[key])
            else:
                data[key][age_group] = self.preprocess_data(data[key][age_group], ground_truth_colum)
                data[key][age_group]['predicted_flow'] = self.fitted_model.predict(data[key][age_group])

        return data