from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "src" / "simulation" / "output" / "paper_simulation.csv",
    PROJECT_ROOT / "src" / "simulation" / "output" / "all_simulation_results.csv",
]

AREA_LABELS = {
    "316": "Mistelbach",
    "314": "Lilienfeld",
    "502": "Hallein",
}

MODEL_LABELS = {
    "delta_gravity": "Gravity Model",
    "delta_ensemble": "AutoGluon",
    "delta_xg": "Constrained XGBoost",
    "delta_cb": "Constrained Catboost",
    "delta_amount": "Ground Truth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render simulation comparison plots as SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Defaults to paper_simulation.csv if available, otherwise all_simulation_results.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "src" / "simulation" / "output" / "simulation_plot",
        help="Output path prefix. Two files will be created with _models_1.svg and _models_2.svg suffixes.",
    )
    return parser.parse_args()


def resolve_input_path(candidate: Path | None) -> Path:
    if candidate is not None:
        return candidate.resolve()
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No simulation CSV found. Expected paper_simulation.csv or all_simulation_results.csv.")


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["delta_gravity", "delta_ensemble", "delta_amount", "delta_xg", "delta_cb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["area_code"] = df["area_code"].astype(str).map(lambda code: AREA_LABELS.get(code, code))
    df["scenario_name"] = df["scenario_name"].astype(str)
    return df


def build_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_long = df.melt(
        id_vars=["year", "area_code", "scenario_name"],
        value_vars=["delta_gravity", "delta_ensemble", "delta_amount", "delta_xg", "delta_cb"],
        var_name="model",
        value_name="delta_value",
    )
    df_long = df_long[df_long["year"].notna() & df_long["delta_value"].notna()].copy()
    df_long["model"] = df_long["model"].map(MODEL_LABELS)
    return df_long


def build_scenario_colors(scenarios: Iterable[str]) -> dict[str, str]:
    scenarios = list(dict.fromkeys(scenarios))
    non_original = [scenario for scenario in scenarios if scenario != "original"]
    colors: dict[str, str] = {"original": "#000000"}
    if non_original:
        total = max(1, len(non_original))
        for index, scenario in enumerate(non_original):
            hue = index / total
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.88)
            colors[scenario] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
    return colors


def scenario_stroke_widths(scenarios: Iterable[str]) -> dict[str, float]:
    return {scenario: (2.6 if scenario == "original" else 1.2) for scenario in scenarios}


def nice_step(value_range: float) -> float:
    if value_range <= 0:
        return 1.0
    exponent = math.floor(math.log10(value_range))
    fraction = value_range / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 0.2
    elif fraction <= 2:
        nice_fraction = 0.5
    elif fraction <= 5:
        nice_fraction = 1
    else:
        nice_fraction = 2
    return nice_fraction * (10 ** exponent)


def compute_ticks(y_min: float, y_max: float, target_ticks: int = 5) -> list[float]:
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    value_range = y_max - y_min
    step = nice_step(value_range / max(target_ticks - 1, 1))
    tick_start = math.floor(y_min / step) * step
    tick_end = math.ceil(y_max / step) * step

    ticks = []
    current = tick_start
    while current <= tick_end + step * 0.5:
        ticks.append(round(current, 10))
        current += step
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def polyline_points(
    years: list[int],
    values: list[float],
    x_positions: dict[int, float],
    y_scale,
) -> str:
    points = []
    for year, value in zip(years, values):
        points.append(f"{x_positions[int(year)]:.2f},{y_scale(float(value)):.2f}")
    return " ".join(points)


class PdfCanvas:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def _y(self, value: float) -> float:
        return self.height - value

    def _escape(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _color(self, hex_color: str) -> tuple[float, float, float]:
        named_colors = {
            "white": "#ffffff",
            "black": "#000000",
            "none": "#ffffff",
        }
        hex_color = named_colors.get(hex_color.lower(), hex_color)
        color = hex_color.lstrip("#")
        return tuple(int(color[index:index + 2], 16) / 255 for index in (0, 2, 4))

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float, dash: str | None = None) -> None:
        r, g, b = self._color(color)
        dash_cmd = "[] 0 d" if not dash else f"[{' '.join(dash.split())}] 0 d"
        self.commands.append(
            f"q {r:.4f} {g:.4f} {b:.4f} RG {width:.2f} w {dash_cmd} {x1:.2f} {self._y(y1):.2f} m {x2:.2f} {self._y(y2):.2f} l S Q"
        )

    def rect(self, x: float, y: float, width: float, height: float, stroke: str, stroke_width: float, fill: str | None = None) -> None:
        r_s, g_s, b_s = self._color(stroke)
        fill_cmd = ""
        paint = "S"
        if fill and fill.lower() != "none":
            r_f, g_f, b_f = self._color(fill)
            fill_cmd = f"{r_f:.4f} {g_f:.4f} {b_f:.4f} rg "
            paint = "B"
        self.commands.append(
            f"q {r_s:.4f} {g_s:.4f} {b_s:.4f} RG {fill_cmd}{stroke_width:.2f} w {x:.2f} {self._y(y) - height:.2f} {width:.2f} {height:.2f} re {paint} Q"
        )

    def circle(self, cx: float, cy: float, r: float, fill: str) -> None:
        k = 0.552284749831 * r
        x0, y0 = cx - r, self._y(cy)
        x1, y1 = cx, self._y(cy - r)
        x2, y2 = cx + r, self._y(cy)
        x3, y3 = cx, self._y(cy + r)
        red, green, blue = self._color(fill)
        self.commands.append(
            "q "
            f"{red:.4f} {green:.4f} {blue:.4f} rg "
            f"{x0:.2f} {y0:.2f} m "
            f"{x0:.2f} {y0 + k:.2f} {x1 - k:.2f} {y1:.2f} {x1:.2f} {y1:.2f} c "
            f"{x1 + k:.2f} {y1:.2f} {x2:.2f} {y2 + k:.2f} {x2:.2f} {y2:.2f} c "
            f"{x2:.2f} {y2 - k:.2f} {x3 + k:.2f} {y3:.2f} {x3:.2f} {y3:.2f} c "
            f"{x3 - k:.2f} {y3:.2f} {x0:.2f} {y0 - k:.2f} {x0:.2f} {y0:.2f} c f Q"
        )

    def polyline(self, points: list[tuple[float, float]], color: str, width: float, dash: str | None = None) -> None:
        if len(points) < 2:
            return
        r, g, b = self._color(color)
        dash_cmd = "[] 0 d" if not dash else f"[{' '.join(dash.split())}] 0 d"
        segments = [f"{points[0][0]:.2f} {self._y(points[0][1]):.2f} m"]
        for x, y in points[1:]:
            segments.append(f"{x:.2f} {self._y(y):.2f} l")
        self.commands.append(
            f"q {r:.4f} {g:.4f} {b:.4f} RG {width:.2f} w {dash_cmd} " + " ".join(segments) + " S Q"
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        anchor: str = "start",
        bold: bool = False,
        rotate: bool = False,
    ) -> None:
        font = "/F2" if bold else "/F1"
        approx_width = len(value) * size * 0.52
        tx = x
        if anchor == "middle":
            tx -= approx_width / 2
        elif anchor == "end":
            tx -= approx_width
        ty = self._y(y)
        escaped = self._escape(value)
        if rotate:
            self.commands.append(
                f"BT 0 1 -1 0 {tx:.2f} {ty:.2f} Tm {font} {size:.2f} Tf ({escaped}) Tj ET"
            )
        else:
            self.commands.append(
                f"BT 1 0 0 1 {tx:.2f} {ty:.2f} Tm {font} {size:.2f} Tf ({escaped}) Tj ET"
            )

    def save(self, path: Path) -> None:
        content = "\n".join(self.commands).encode("latin-1", errors="replace")
        objects: list[bytes] = []

        def add_object(data: bytes) -> int:
            objects.append(data)
            return len(objects)

        font1 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font2 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        content_obj = add_object(f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream")
        page_obj = add_object(
            f"<< /Type /Page /Parent 5 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
            f"/Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> >> /Contents {content_obj} 0 R >>".encode("latin-1")
        )
        pages_obj = add_object(f"<< /Type /Pages /Kids [{page_obj} 0 R] /Count 1 >>".encode("latin-1"))
        catalog_obj = add_object(b"<< /Type /Catalog /Pages 5 0 R >>")

        offsets = [0]
        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
        pdf.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
        )
        path.write_bytes(pdf)


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    width = float(root.attrib["width"])
    height = float(root.attrib["height"])
    canvas = PdfCanvas(width, height)

    def parse_svg_length(value: str, total: float) -> float:
        if value.endswith("%"):
            return total * float(value[:-1]) / 100
        return float(value)

    ns_suffix = "}"
    for element in root:
        tag = element.tag.split(ns_suffix)[-1]
        if tag == "rect":
            canvas.rect(
                x=parse_svg_length(element.attrib["x"], width) if "x" in element.attrib else 0.0,
                y=parse_svg_length(element.attrib["y"], height) if "y" in element.attrib else 0.0,
                width=parse_svg_length(element.attrib["width"], width),
                height=parse_svg_length(element.attrib["height"], height),
                stroke=element.attrib.get("stroke", "#000000"),
                stroke_width=float(element.attrib.get("stroke-width", "1")),
                fill=element.attrib.get("fill"),
            )
        elif tag == "line":
            canvas.line(
                x1=float(element.attrib["x1"]),
                y1=float(element.attrib["y1"]),
                x2=float(element.attrib["x2"]),
                y2=float(element.attrib["y2"]),
                color=element.attrib.get("stroke", "#000000"),
                width=float(element.attrib.get("stroke-width", "1")),
                dash=element.attrib.get("stroke-dasharray"),
            )
        elif tag == "circle":
            canvas.circle(
                cx=float(element.attrib["cx"]),
                cy=float(element.attrib["cy"]),
                r=float(element.attrib["r"]),
                fill=element.attrib.get("fill", "#000000"),
            )
        elif tag == "polyline":
            points = []
            for pair in element.attrib["points"].split():
                x_val, y_val = pair.split(",")
                points.append((float(x_val), float(y_val)))
            canvas.polyline(
                points=points,
                color=element.attrib.get("stroke", "#000000"),
                width=float(element.attrib.get("stroke-width", "1")),
                dash=element.attrib.get("stroke-dasharray"),
            )
        elif tag == "text":
            transform = element.attrib.get("transform", "")
            canvas.text(
                x=float(element.attrib.get("x", "0")),
                y=float(element.attrib.get("y", "0")),
                value=element.text or "",
                size=float(element.attrib.get("font-size", "12")),
                anchor=element.attrib.get("text-anchor", "start"),
                bold=element.attrib.get("font-weight") == "bold",
                rotate="rotate(" in transform,
            )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(pdf_path)


def render_plot(
    data: pd.DataFrame,
    gt_data: pd.DataFrame,
    models: list[str],
    output_path: Path,
    scenario_colors: dict[str, str],
    scenario_sizes: dict[str, float],
) -> None:
    filtered = data[data["model"].isin(models)].copy()
    filtered = filtered[~((filtered["model"] == "Ground Truth") & (filtered["scenario_name"] != "original"))]

    areas = list(dict.fromkeys(filtered["area_code"].tolist()))
    years = sorted(int(year) for year in filtered["year"].dropna().unique())
    gt_subset = gt_data[gt_data["area_code"].isin(areas)].copy()

    facet_width = 360
    facet_height = 240
    margin_left = 90
    margin_right = 260
    margin_top = 90
    margin_bottom = 70
    plot_left_pad = 55
    plot_right_pad = 16
    plot_top_pad = 26
    plot_bottom_pad = 42
    facet_gap_x = 20
    facet_gap_y = 20

    width = margin_left + margin_right + len(models) * facet_width + (len(models) - 1) * facet_gap_x
    height = margin_top + margin_bottom + len(areas) * facet_height + (len(areas) - 1) * facet_gap_y + 80

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" font-size="24" text-anchor="middle" font-family="Arial">Simulation comparison</text>',
        f'<text x="{width / 2:.1f}" y="{height - 18}" font-size="18" text-anchor="middle" font-family="Arial">Year</text>',
        f'<text x="24" y="{height / 2:.1f}" font-size="18" text-anchor="middle" transform="rotate(-90 24 {height / 2:.1f})" font-family="Arial">Net Migration</text>',
    ]

    for row_index, area in enumerate(areas):
        y_offset = margin_top + row_index * (facet_height + facet_gap_y)
        svg_parts.append(
            f'<text x="12" y="{y_offset + facet_height / 2:.1f}" font-size="18" font-weight="bold" '
            f'text-anchor="start" dominant-baseline="middle" font-family="Arial">{escape(str(area))}</text>'
        )
        for col_index, model in enumerate(models):
            x_offset = margin_left + col_index * (facet_width + facet_gap_x)
            if row_index == 0:
                svg_parts.append(
                    f'<text x="{x_offset + facet_width / 2:.1f}" y="{margin_top - 30}" font-size="18" '
                    f'font-weight="bold" text-anchor="middle" font-family="Arial">{escape(model)}</text>'
                )

            facet_df = filtered[(filtered["area_code"] == area) & (filtered["model"] == model)].copy()
            facet_gt = gt_subset[gt_subset["area_code"] == area].copy()
            combined_values = facet_df["delta_value"].tolist() + facet_gt["delta_value"].tolist()
            if not combined_values:
                continue

            ticks = compute_ticks(min(combined_values), max(combined_values))
            y_min = ticks[0]
            y_max = ticks[-1]
            plot_x0 = x_offset + plot_left_pad
            plot_y0 = y_offset + plot_top_pad
            plot_width = facet_width - plot_left_pad - plot_right_pad
            plot_height = facet_height - plot_top_pad - plot_bottom_pad

            if len(years) == 1:
                x_positions = {years[0]: plot_x0 + plot_width / 2}
            else:
                span = years[-1] - years[0]
                x_positions = {
                    year: plot_x0 + ((year - years[0]) / span) * plot_width
                    for year in years
                }

            def y_scale(value: float) -> float:
                if y_max == y_min:
                    return plot_y0 + plot_height / 2
                return plot_y0 + (1 - (value - y_min) / (y_max - y_min)) * plot_height

            svg_parts.append(
                f'<rect x="{x_offset}" y="{y_offset}" width="{facet_width}" height="{facet_height}" '
                f'fill="none" stroke="#d7d7d7" stroke-width="1"/>'
            )

            for tick in ticks:
                y = y_scale(float(tick))
                svg_parts.append(
                    f'<line x1="{plot_x0}" y1="{y:.2f}" x2="{plot_x0 + plot_width}" y2="{y:.2f}" '
                    f'stroke="#ececec" stroke-width="1"/>'
                )
                svg_parts.append(
                    f'<text x="{plot_x0 - 8}" y="{y + 5:.2f}" font-size="12" text-anchor="end" font-family="Arial">{fmt_tick(tick)}</text>'
                )

            for year in years:
                x = x_positions[year]
                svg_parts.append(
                    f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y0 + plot_height}" stroke="#f5f5f5" stroke-width="1"/>'
                )
                svg_parts.append(
                    f'<text x="{x:.2f}" y="{plot_y0 + plot_height + 22}" font-size="12" text-anchor="middle" font-family="Arial">{year}</text>'
                )

            for scenario_name, scenario_df in facet_df.groupby("scenario_name", sort=False):
                scenario_df = scenario_df.sort_values("year")
                points = polyline_points(
                    scenario_df["year"].astype(int).tolist(),
                    scenario_df["delta_value"].astype(float).tolist(),
                    x_positions,
                    y_scale,
                )
                color = scenario_colors[scenario_name]
                width_px = scenario_sizes[scenario_name]
                svg_parts.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width_px}" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>'
                )
                for year, value in zip(scenario_df["year"], scenario_df["delta_value"]):
                    svg_parts.append(
                        f'<circle cx="{x_positions[int(year)]:.2f}" cy="{y_scale(float(value)):.2f}" r="3.2" fill="{color}"/>'
                    )

            gt_model = facet_gt.copy().sort_values("year")
            if not gt_model.empty:
                points = polyline_points(
                    gt_model["year"].astype(int).tolist(),
                    gt_model["delta_value"].astype(float).tolist(),
                    x_positions,
                    y_scale,
                )
                svg_parts.append(
                    f'<polyline points="{points}" fill="none" stroke="#000000" stroke-width="2.2" '
                    f'stroke-dasharray="8 6" stroke-linejoin="round" stroke-linecap="round"/>'
                )

    legend_x = width - margin_right + 20
    legend_y = margin_top + 10
    legend_row_height = 24
    svg_parts.append(f'<text x="{legend_x}" y="{legend_y}" font-size="15" font-family="Arial">Scenario</text>')
    for index, scenario_name in enumerate(scenario_colors):
        y = legend_y + 22 + index * legend_row_height
        color = scenario_colors[scenario_name]
        stroke_width = scenario_sizes[scenario_name]
        svg_parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
        svg_parts.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="3.2" fill="{color}"/>')
        svg_parts.append(
            f'<text x="{legend_x + 38}" y="{y + 4}" font-size="13" font-family="Arial">{escape(str(scenario_name))}</text>'
        )

    gt_y = legend_y + 22 + len(scenario_colors) * legend_row_height + 12
    svg_parts.append(f'<text x="{legend_x}" y="{gt_y}" font-size="15" font-family="Arial">Reference</text>')
    gt_line_y = gt_y + 18
    svg_parts.append(
        f'<line x1="{legend_x}" y1="{gt_line_y}" x2="{legend_x + 28}" y2="{gt_line_y}" stroke="#000000" '
        f'stroke-width="2.2" stroke-dasharray="8 6"/>'
    )
    svg_parts.append(
        f'<text x="{legend_x + 38}" y="{gt_line_y + 4}" font-size="13" font-family="Arial">Ground Truth</text>'
    )

    svg_parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_prefix = args.output_prefix.resolve()

    df = load_data(input_path)
    df_long = build_long_frame(df)

    scenarios = list(dict.fromkeys(df_long["scenario_name"].tolist()))
    scenario_colors = build_scenario_colors(scenarios)
    scenario_sizes = scenario_stroke_widths(scenarios)
    gt_data = df_long[(df_long["model"] == "Ground Truth") & (df_long["scenario_name"] == "original")].copy()

    render_plot(
        data=df_long,
        gt_data=gt_data,
        models=["Gravity Model", "AutoGluon"],
        output_path=output_prefix.with_name(f"{output_prefix.name}_models_1.svg"),
        scenario_colors=scenario_colors,
        scenario_sizes=scenario_sizes,
    )
    render_plot(
        data=df_long,
        gt_data=gt_data,
        models=["Constrained XGBoost", "Constrained Catboost"],
        output_path=output_prefix.with_name(f"{output_prefix.name}_models_2.svg"),
        scenario_colors=scenario_colors,
        scenario_sizes=scenario_sizes,
    )

    svg_to_pdf(
        output_prefix.with_name(f"{output_prefix.name}_models_1.svg"),
        output_prefix.with_name(f"{output_prefix.name}_models_1.pdf"),
    )
    svg_to_pdf(
        output_prefix.with_name(f"{output_prefix.name}_models_2.svg"),
        output_prefix.with_name(f"{output_prefix.name}_models_2.pdf"),
    )

    print(f"Wrote plot SVG and PDF files next to {output_prefix}")


if __name__ == "__main__":
    main()
