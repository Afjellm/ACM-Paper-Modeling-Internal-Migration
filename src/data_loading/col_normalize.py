import pandas as pd


def norm_count_cols(df: pd.DataFrame) -> pd.DataFrame:
    postfixes = ["target", "origin"]
    cols_to_normalize = ["schools", "c_500", "c_1000", "c_50_99", "c_10_19", "c_20_49", "c_250_499",
                         "c_100_250", "w_permit_11", "w_permit_3_10", "w_permit_1_2"]

    df = df.copy()

    for pfx in postfixes:
        for col in cols_to_normalize:
            col_name = f"{col}_{pfx}"
            pop_col_name = f"population_total_{pfx}"
            if col_name in df.columns:
                df[col_name] = df[col_name] / df[pop_col_name] * 1000

    return df