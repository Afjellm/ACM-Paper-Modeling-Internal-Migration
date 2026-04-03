from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


VALIDATION_PREDICT_YEARS: dict[str, set[int]] = {
    "2018_2019_2020_2021": {2022, 2023},
    "2019_2020_2021_2022": {2018, 2023},
    "2020_2021_2022_2023": {2018, 2019},
    "2018_2019_2020": {2021, 2022},
    "2019_2020_2021": {2022, 2023},
    "2020_2021_2022": {2019, 2023},
    "2021_2022_2023": {2019, 2020},
}

# Configure the four model families here.
MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "xgboost": {
        "folder": r"models/output/xgboost_output/xboost_model_s1_normalized_cols_amount_next_year_rows_1000",
        "fit_year": "2020_2021_2022",
    },
    "constrained_catboost": {
        "folder": r"models/output/catboost_output/caboost_s1_normalized_cols_amount_rows_1000",
        "fit_year": "2020_2021_2022_2023",
    },
    "autogluon": {
        "folder": r"models/output/autogluon_output/autogluon_amount_rows_1000",
        "fit_year": "2019_2020_2021_2022",
    },
    "gravity": {
        "folder": r"models/output/gravity_output/gravity_prev_year_age_pop_rows_1000",
        "fit_year": "2019_2020_2021_2022",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the configured XGBoost, CatBoost, AutoGluon, and Gravity outputs."
    )
    parser.add_argument(
        "--output",
        default="./model_comparison_analysis.md",
        help="Markdown output path for the comparison report.",
    )
    return parser.parse_args()


def read_prediction_csv(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode '{csv_path}' with utf-8, cp1252, or latin1.",
    )


def load_raw_predictions(model_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in model_dir.rglob("full_results.csv"):
        rel_parts = csv_path.relative_to(model_dir).parts
        if len(rel_parts) < 3:
            continue

        fit_year, predict_year_raw = rel_parts[:2]
        frame = read_prediction_csv(csv_path).assign(
            fit_year=fit_year,
            predict_year=int(predict_year_raw),
        )
        if len(rel_parts) >= 4 and "age_group" not in frame.columns:
            frame["age_group"] = rel_parts[2]

        if {"amount_next_year", "predicted_flow"}.issubset(frame.columns):
            frame = frame.rename(
                columns={
                    "amount_next_year": "amount",
                    "predicted_flow": "predicted",
                }
            )

        if {"amount", "predicted", "area_code_origin", "area_code_target", "age_group"}.issubset(frame.columns):
            frame["predicted"] = frame["predicted"].round()
            frames.append(
                frame[
                    [
                        "fit_year",
                        "predict_year",
                        "age_group",
                        "area_code_origin",
                        "area_code_target",
                        "amount",
                        "predicted",
                    ]
                ].copy()
            )

    if not frames:
        raise FileNotFoundError(f"No usable 'full_results.csv' files found under '{model_dir}'.")

    return pd.concat(frames, ignore_index=True)


def raw_to_age_aggregated(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    detailed = raw_predictions.loc[
        raw_predictions["area_code_origin"] != raw_predictions["area_code_target"]
    ].copy()
    in_agg = (
        detailed.groupby(["fit_year", "predict_year", "age_group", "area_code_target"], as_index=False)
        .agg(amount_in=("amount", "sum"), predicted_in=("predicted", "sum"))
        .rename(columns={"area_code_target": "area_code"})
    )
    out_agg = (
        detailed.groupby(["fit_year", "predict_year", "age_group", "area_code_origin"], as_index=False)
        .agg(amount_out=("amount", "sum"), predicted_out=("predicted", "sum"))
        .rename(columns={"area_code_origin": "area_code"})
    )
    return in_agg.merge(
        out_agg,
        on=["fit_year", "predict_year", "age_group", "area_code"],
        how="outer",
    ).fillna(0)


def filter_validation_set(data: pd.DataFrame, selected_fit_year: str) -> pd.DataFrame:
    valid_fit_years = data["fit_year"].isin(VALIDATION_PREDICT_YEARS)
    valid_predict_years = data.apply(
        lambda row: row["predict_year"] in VALIDATION_PREDICT_YEARS.get(row["fit_year"], set()),
        axis=1,
    )
    filtered = data.loc[valid_fit_years & valid_predict_years].copy()
    filtered = filtered.loc[filtered["fit_year"] == selected_fit_year].copy()
    return filtered


def safe_mare(grouped: pd.DataFrame, actual_col: str, predicted_col: str, output_col: str) -> pd.DataFrame:
    grouped = grouped.copy()
    grouped = grouped.loc[grouped[actual_col] != 0].copy()
    grouped[output_col] = (grouped[predicted_col] - grouped[actual_col]).abs() / grouped[actual_col]
    return grouped


def compute_mare(age_aggregated: pd.DataFrame) -> tuple[float, float]:
    in_grouped = (
        age_aggregated.groupby("area_code", as_index=False)
        .agg(amount_in=("amount_in", "sum"), predicted_in=("predicted_in", "sum"))
    )
    in_grouped = safe_mare(in_grouped, "amount_in", "predicted_in", "mare_in")
    mare_in = in_grouped["mare_in"].mean()

    out_grouped = (
        age_aggregated.groupby("area_code", as_index=False)
        .agg(amount_out=("amount_out", "sum"), predicted_out=("predicted_out", "sum"))
    )
    out_grouped = safe_mare(out_grouped, "amount_out", "predicted_out", "mare_out")
    mare_out = out_grouped["mare_out"].mean()
    return mare_in, mare_out


def sign_value(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_constrained_diff(age_aggregated: pd.DataFrame) -> tuple[int, int, float]:
    totals = (
        age_aggregated.groupby("area_code", as_index=False)
        .agg(
            sum_in=("amount_in", "sum"),
            sum_out=("amount_out", "sum"),
            predicted_in=("predicted_in", "sum"),
            predicted_out=("predicted_out", "sum"),
        )
    )
    totals["delta_truth"] = totals["sum_in"] - totals["sum_out"]
    totals["delta_predicted"] = totals["predicted_in"] - totals["predicted_out"]
    totals["sign_diff"] = totals["delta_truth"].apply(sign_value) != totals["delta_predicted"].apply(sign_value)

    constrained_diff = int(totals["sign_diff"].sum())
    total_areas = int(totals["area_code"].nunique())
    share = constrained_diff / total_areas if total_areas else float("nan")
    return constrained_diff, total_areas, share


def r2_score(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    ss_res = ((predicted - actual) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def summarize_r2(age_aggregated: pd.DataFrame, actual_col: str, predicted_col: str) -> tuple[float, float, str, float, str]:
    age_scores = (
        age_aggregated.groupby("age_group")
        .apply(lambda group: r2_score(group[actual_col], group[predicted_col]), include_groups=False)
        .reset_index(name="r2")
    )
    valid = age_scores.dropna(subset=["r2"]).copy()
    if valid.empty:
        return float("nan"), float("nan"), "n/a", float("nan"), "n/a"

    min_idx = valid["r2"].idxmin()
    max_idx = valid["r2"].idxmax()
    return (
        valid["r2"].mean(),
        valid.loc[min_idx, "r2"],
        valid.loc[min_idx, "age_group"],
        valid.loc[max_idx, "r2"],
        valid.loc[max_idx, "age_group"],
    )


def compute_cpc_by_age(raw_predictions: pd.DataFrame, model_label: str) -> pd.DataFrame:
    cpc = (
        raw_predictions.groupby("age_group", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "cpc": (
                        2 * group[["amount", "predicted"]].min(axis=1).sum()
                    ) / (group["amount"].sum() + group["predicted"].sum())
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    cpc["model"] = model_label
    return cpc[["age_group", "cpc", "model"]]


def age_group_sort_key(age_group: str) -> float:
    digits = "".join(character for character in age_group if character.isdigit() or character == ",")
    if not digits:
        return float("inf")
    return float(digits.split(",")[0])


def plot_cpc(cpc_df: pd.DataFrame, output_path: Path) -> None:
    age_levels = sorted(cpc_df["age_group"].dropna().unique(), key=age_group_sort_key)
    cpc_df = cpc_df.copy()
    cpc_df["age_group"] = pd.Categorical(cpc_df["age_group"], categories=age_levels, ordered=True)
    cpc_df = cpc_df.sort_values(["model", "age_group"])
    colors = {
        "XGBoost": "#1f77b4",
        "CatBoost": "#d62728",
        "AutoGluon": "#2ca02c",
        "Gravity": "#ff7f0e",
    }

    width = 1400
    height = 800
    margin_left = 90
    margin_right = 40
    margin_top = 40
    margin_bottom = 220
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_position(index: int) -> float:
        if len(age_levels) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(age_levels) - 1))

    def y_position(value: float) -> float:
        return margin_top + plot_height * (1 - value)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-size="24" font-family="Arial">Age group</text>',
        (
            f'<text x="28" y="{margin_top + plot_height / 2}" text-anchor="middle" '
            'font-size="24" font-family="Arial" transform="rotate(-90 28 '
            f'{margin_top + plot_height / 2})">CPC</text>'
        ),
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_position(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 6}" text-anchor="end" font-size="18" font-family="Arial">{tick:.2f}</text>'
        )

    for index, age_group in enumerate(age_levels):
        x = x_position(index)
        svg_parts.append(
            f'<text x="{x}" y="{margin_top + plot_height + 18}" text-anchor="end" '
            f'font-size="18" font-family="Arial" transform="rotate(-45 {x} {margin_top + plot_height + 18})">'
            f'{html.escape(str(age_group))}</text>'
        )

    legend_x = margin_left
    legend_y = height - 95
    for model_index, model_name in enumerate(cpc_df["model"].unique()):
        color = colors.get(model_name, "#444")
        x = legend_x + model_index * 250
        svg_parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg_parts.append(f'<circle cx="{x + 14}" cy="{legend_y}" r="4" fill="{color}"/>')
        svg_parts.append(
            f'<text x="{x + 38}" y="{legend_y + 6}" font-size="20" font-family="Arial">{html.escape(model_name)}</text>'
        )

    for model_name, group in cpc_df.groupby("model"):
        color = colors.get(model_name, "#444")
        points: list[str] = []
        for _, row in group.iterrows():
            age_index = age_levels.index(row["age_group"])
            x = x_position(age_index)
            y = y_position(float(row["cpc"]))
            points.append(f"{x},{y}")
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')

        if points:
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>'
            )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def analyze_model(model_name: str, config: dict[str, str]) -> dict[str, object]:
    model_dir = Path(config["folder"]).resolve()
    fit_year = config["fit_year"]
    model_label = {
        "xgboost": "XGBoost",
        "constrained_catboost": "CatBoost",
        "autogluon": "AutoGluon",
        "gravity": "Gravity",
    }[model_name]
    raw_predictions = load_raw_predictions(model_dir)
    filtered_raw = filter_validation_set(raw_predictions, fit_year)
    validation_df = raw_to_age_aggregated(filtered_raw)

    if validation_df.empty:
        raise ValueError(
            f"Validation filter returned no rows for model '{model_name}' and fit_year '{fit_year}'."
        )

    mare_in, mare_out = compute_mare(validation_df)
    constrained_diff, total_areas, constrained_share = compute_constrained_diff(validation_df)

    r2_in, min_r2_in, min_r2_in_age, max_r2_in, max_r2_in_age = summarize_r2(
        validation_df, "amount_in", "predicted_in"
    )
    r2_out, min_r2_out, min_r2_out_age, max_r2_out, max_r2_out_age = summarize_r2(
        validation_df, "amount_out", "predicted_out"
    )
    cpc_by_age = compute_cpc_by_age(filtered_raw, model_label)

    return {
        "model": model_name,
        "model_label": model_label,
        "folder": str(model_dir),
        "fit_year": fit_year,
        "mare_in": mare_in,
        "mare_out": mare_out,
        "constrained_diff": constrained_diff,
        "total_areas": total_areas,
        "constrained_share": constrained_share,
        "r2_in": r2_in,
        "min_r2_in": min_r2_in,
        "min_r2_in_age_group": min_r2_in_age,
        "max_r2_in": max_r2_in,
        "max_r2_in_age_group": max_r2_in_age,
        "r2_out": r2_out,
        "min_r2_out": min_r2_out,
        "min_r2_out_age_group": min_r2_out_age,
        "max_r2_out": max_r2_out,
        "max_r2_out_age_group": max_r2_out_age,
        "cpc_by_age": cpc_by_age,
    }


def format_float(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.6f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(format_float)

    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in formatted.astype(object).itertuples(index=False, name=None):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_report(results_df: pd.DataFrame, cpc_image_path: Path) -> str:
    model_overview = results_df[["model", "fit_year", "folder"]].copy()
    mare_table = results_df[["model", "mare_in", "mare_out"]].copy()
    constrained_table = pd.DataFrame(
        {
            "model": results_df["model"],
            "correctly predicted (S_agree)": (results_df["total_areas"] - results_df["constrained_diff"]).astype(str)
            + "/"
            + results_df["total_areas"].astype(str),
        }
    )
    r2_in_table = results_df[
        ["model", "r2_in", "min_r2_in", "min_r2_in_age_group", "max_r2_in", "max_r2_in_age_group"]
    ].copy()
    r2_out_table = results_df[
        ["model", "r2_out", "min_r2_out", "min_r2_out_age_group", "max_r2_out", "max_r2_out_age_group"]
    ].copy()

    sections = [
        "# Model Comparison Analysis",
        "## Models",
        dataframe_to_markdown(model_overview),
        "## MARE",
        dataframe_to_markdown(mare_table),
        "## Number of incorrect predicted districts",
        dataframe_to_markdown(constrained_table),
        "## R2 In Migration",
        dataframe_to_markdown(r2_in_table),
        "## R2 Out Migration",
        dataframe_to_markdown(r2_out_table),
        "## CPC By Age",
        f"![CPC by age]({cpc_image_path.name})",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpc_plot_path = output_path.with_name(f"{output_path.stem}_cpc.svg")

    results = [analyze_model(model_name, config) for model_name, config in MODEL_CONFIGS.items()]
    cpc_frames = [result.pop("cpc_by_age") for result in results]
    results_df = pd.DataFrame(results)

    cpc_df = pd.concat(cpc_frames, ignore_index=True)
    plot_cpc(cpc_df, cpc_plot_path)
    output_path.write_text(build_report(results_df, cpc_plot_path), encoding="utf-8")
    print(f"Wrote analysis report to {output_path}")


if __name__ == "__main__":
    main()
