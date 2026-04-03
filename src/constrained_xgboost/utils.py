from os import PathLike

import pandas as pd


def _parse_sign(cell):
    """Map Excel symbols/values to {-1, 0, 1}."""
    if pd.isna(cell):
        return 0
    s = str(cell).strip().lower()
    if s in {"+", "+1", "1", "plus"}: return 1
    if s in {"-", "-1", "minus"}:     return -1
    if s in {"0", "", "none", "na", "n/a"}: return 0
    # Fallback: numeric strings (incl. EU decimal)
    try:
        v = float(s.replace(",", "."))
        return 1 if v > 0 else (-1 if v < 0 else 0)
    except Exception:
        raise ValueError(f"Cannot parse monotone sign from '{cell}'")


def load_monotone_constraints_from_xlsx(
        xlsx_path: str | PathLike,
        age_group_value,
        features_for_training_numeric,
        sheet_name=0,
        age_col="age_group"
):
    """
    Excel layout example:
      age_group | c_500_target | c_1000_origin | c_1000_target | ...
           0,5 |           +             |            -            |            +            | ...

    Returns a tuple aligned with features_for_training_numeric.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    def _norm_age(v):
        if pd.isna(v): return None
        s = str(v).strip()
        return s.replace(",", ".")

    needle = _norm_age(age_group_value)

    df["_age_norm"] = df[age_col].map(_norm_age)
    row = df.loc[df["_age_norm"] == needle]

    if row.empty:
        raise KeyError(
            f"Age group '{age_group_value}' not found in '{age_col}'. "
            f"Available: {sorted(set(df['_age_norm'].dropna()))}"
        )
    if len(row) > 1:
        raise ValueError(f"Multiple rows matched age group '{age_group_value}'. Please make it unique.")

    row = row.iloc[0]

    constraints_from_file = {
        col: _parse_sign(val)
        for col, val in row.items()
        if col not in {age_col, "_age_norm"}
    }

    missing = [f for f in features_for_training_numeric if f not in constraints_from_file]
    if missing:
        print(
            f"[monotone] no monotone constraint configuration found for  {len(missing)} features; defaulting to 0 (unconstrained): {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    monotone_constraints = tuple(constraints_from_file.get(f, 0) for f in features_for_training_numeric)

    bad = [v for v in monotone_constraints if v not in (-1, 0, 1)]
    if bad:
        raise ValueError(f"Invalid monotone values present: {set(bad)}")

    return monotone_constraints
