"""
Native Excel chart and advanced-formatting library.

Everything here produces **real Excel objects** — live charts bound to worksheet
ranges, AutoFilters, data-validation dropdowns, data bars, defined names and
print setup. Nothing is a pasted image.

Why that distinction matters: a chart image is dead the moment the underlying
number changes, and an executive who filters a table expects the chart above it
to follow. Charts built on worksheet references stay live, recalculate when a
cell is edited, and survive the workbook being emailed to someone who has never
heard of Python. That is the difference between exporting a report and
delivering a tool.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from openpyxl.chart import (BarChart, LineChart, PieChart, Reference, ScatterChart,
                            Series)
from openpyxl.chart.axis import ChartLines
from openpyxl.chart.data_source import (AxDataSource, NumDataSource, NumRef,
                                        StrRef)
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.marker import Marker
from openpyxl.chart.trendline import Trendline
from openpyxl.drawing.line import LineProperties
from openpyxl.formatting.rule import DataBarRule, IconSetRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.worksheet.worksheet import Worksheet

# --------------------------------------------------------------------------- #
# Palette (hex without the leading '#', as openpyxl expects)
# --------------------------------------------------------------------------- #
NAVY = "1F3864"
BLUE = "2E75B6"
LIGHT = "D9E2F3"
GREEN = "0F7B4F"
AMBER = "B7791F"
RED = "9C2B2B"
GREY = "808080"

SERIES_COLOURS = [NAVY, BLUE, GREEN, AMBER, RED, "7030A0", "00838F", "5B6770"]


def place_chart(ws: Worksheet, chart, anchor: str) -> None:
    """
    Attach a chart to a worksheet with hidden-cell plotting ENABLED.

    Excel's ``plotVisOnly`` flag defaults to true, which means a chart refuses
    to plot any series whose source cells sit in a hidden row or column. Since
    the source blocks for these charts are staged in a hidden far-right column
    band, leaving the default in place renders every chart completely blank —
    the chart object exists, the series are wired correctly, and nothing draws.

    Setting ``visible_cells_only = False`` is therefore not a nicety; it is the
    difference between a chart and an empty frame.
    """
    chart.visible_cells_only = False
    chart.display_blanks = "gap"
    _fix_text_categories(ws, chart)
    ws.add_chart(chart, anchor)


def _fix_text_categories(ws: Worksheet, chart) -> None:
    """
    Rewrite textual category axes from ``numRef`` to ``strRef``.

    ``openpyxl.set_categories`` always emits a numeric reference. When the
    categories are text -- supplier IDs, month labels, aggregation levels --
    Excel is being told to read words as numbers, and the axis renders blank or
    as 1, 2, 3. Detecting the cell type and swapping in a string reference makes
    the labels appear.
    """
    for series in getattr(chart, "series", []) or []:
        cat = getattr(series, "cat", None)
        if cat is None or cat.numRef is None or cat.strRef is not None:
            continue
        ref = cat.numRef.f
        try:
            cells = ws[ref.split("!")[-1].replace("$", "")]
            first = cells[0][0] if isinstance(cells, tuple) else cells
            value = first.value if hasattr(first, "value") else None
        except Exception:
            continue
        if isinstance(value, str):
            series.cat = AxDataSource(strRef=StrRef(f=ref))


def _style_axes(chart, x_title: str = "", y_title: str = "",
                y2_title: Optional[str] = None) -> None:
    """Consistent axis titles, gridlines and fonts across every chart."""
    if x_title:
        chart.x_axis.title = x_title
    if y_title:
        chart.y_axis.title = y_title
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.y_axis.majorGridlines = ChartLines()
    chart.style = 2


def _apply_series_colour(series, colour: str, width: int = 22_000,
                         dashed: bool = False, smooth: bool = False) -> None:
    """Colour a line/bar series, optionally dashed (for limits and targets)."""
    line = LineProperties(solidFill=colour, w=width)
    if dashed:
        line.prstDash = "dash"
    series.graphicalProperties.line = line
    if hasattr(series, "smooth"):
        series.smooth = smooth


# --------------------------------------------------------------------------- #
# Data staging
# --------------------------------------------------------------------------- #
def write_chart_data(ws: Worksheet, df: pd.DataFrame, start_row: int,
                     start_col: int, title: str = "",
                     hidden: bool = False) -> Tuple[int, int, int, int]:
    """
    Write a compact data block for a chart to reference.

    Returns ``(header_row, first_data_row, last_row, last_col)``.

    Charts must point at cells, so the series data has to live somewhere on a
    sheet. Staging blocks are placed in a far-right column band and optionally
    hidden, keeping the presentation area clean while leaving the numbers
    inspectable by anyone who widens the columns — auditable, not concealed.
    """
    if title:
        cell = ws.cell(row=start_row, column=start_col, value=title)
        cell.font = Font(bold=True, size=9, color=GREY)
        start_row += 1

    for offset, column in enumerate(df.columns):
        cell = ws.cell(row=start_row, column=start_col + offset,
                       value=str(column).replace("_", " "))
        cell.font = Font(bold=True, size=9)

    for r, (_, record) in enumerate(df.iterrows(), start=1):
        for c, column in enumerate(df.columns):
            value = record[column]
            if pd.isna(value):
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            ws.cell(row=start_row + r, column=start_col + c, value=value)

    last_row = start_row + len(df)
    last_col = start_col + len(df.columns) - 1

    # Staging columns are narrowed and greyed rather than hidden. Hidden
    # columns interact badly with chart rendering in several viewers, and the
    # data is meant to be auditable anyway -- it sits far to the right of the
    # presentation area where nobody scrolls by accident.
    if hidden:
        for col in range(start_col, last_col + 1):
            ws.column_dimensions[get_column_letter(col)].width = 11
            ws.column_dimensions[get_column_letter(col)].outlineLevel = 1

    return start_row, start_row + 1, last_row, last_col


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def add_spc_control_chart(ws: Worksheet, df: pd.DataFrame, anchor: str,
                          data_col: int = 40, width: float = 26,
                          height: float = 11) -> None:
    """
    Six Sigma individuals control chart as a live Excel line chart.

    Four series: the OTIF measurement, the centre line, and the upper and lower
    control limits rendered as dashed reference lines. This is the chart a Black
    Belt expects to see, and building it natively means a plant manager can
    extend it next month by typing one more row.
    """
    columns = ["Order_Month", "OTIF_Pct", "Center_Line", "UCL", "LCL"]
    block = df[[c for c in columns if c in df.columns]].copy()
    if block.empty:
        return

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="SPC chart source data", hidden=True)

    chart = LineChart()
    chart.title = "Monthly OTIF — Individuals Control Chart (3σ, moving-range sigma)"
    _style_axes(chart, "Month", "OTIF (%)")
    chart.height, chart.width = height, width

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + 4,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    styles = [(BLUE, False, 28_000), (NAVY, False, 18_000),
              (RED, True, 14_000), (RED, True, 14_000)]
    for series, (colour, dashed, weight) in zip(chart.series, styles):
        _apply_series_colour(series, colour, width=weight, dashed=dashed)
    if chart.series:
        chart.series[0].marker = Marker(symbol="circle", size=7)

    chart.y_axis.scaling.min = max(0, float(block["LCL"].min()) - 4)
    chart.y_axis.scaling.max = min(100, float(block["UCL"].max()) + 3)
    place_chart(ws, chart, anchor)


def add_accuracy_ladder_chart(ws: Worksheet, df: pd.DataFrame, anchor: str,
                              data_col: int = 40, benchmark: float = 15.0) -> None:
    """
    Horizontal bar chart of WAPE by aggregation level, with the benchmark.

    The single most communicative chart in the pack: it shows at a glance that
    forecast error is a function of the level you measure at.
    """
    block = df[["Aggregation_Level", "WAPE"]].copy()
    if block.empty:
        return
    block["Benchmark"] = benchmark

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="Accuracy ladder source data", hidden=True)

    chart = BarChart()
    chart.type = "bar"
    chart.title = f"Forecast Accuracy by Aggregation Level (holdout) — benchmark {benchmark:.0f}%"
    _style_axes(chart, "Aggregation level", "WAPE (%)")
    chart.height, chart.width = 9, 24

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + 2,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showVal = True

    if chart.series:
        chart.series[0].graphicalProperties.solidFill = BLUE
    if len(chart.series) > 1:
        _apply_series_colour(chart.series[1], RED, width=16_000, dashed=True)
        chart.series[1].graphicalProperties.solidFill = None
    place_chart(ws, chart, anchor)


def add_waterfall_chart(ws: Worksheet, waterfall: pd.DataFrame, anchor: str,
                        data_col: int = 46) -> None:
    """
    Consensus bridge as a stacked-bar waterfall.

    Excel's native waterfall type is not exposed by openpyxl, so the classic
    invisible-base technique is used: a transparent spacer series carries each
    floating bar to its correct height. The result is a genuine Excel chart that
    recalculates, not a picture of one.
    """
    if waterfall.empty:
        return

    steps, base, rise, fall = [], [], [], []
    running = 0.0
    for _, row in waterfall.iterrows():
        label, value, measure = row["Step"], float(row["Units"]), row["Measure"]
        if measure == "absolute":
            base.append(0.0); rise.append(value); fall.append(0.0)
            running = value
        elif measure == "total":
            base.append(0.0); rise.append(running); fall.append(0.0)
        elif value >= 0:
            base.append(running); rise.append(value); fall.append(0.0)
            running += value
        else:
            running += value
            base.append(running); rise.append(0.0); fall.append(-value)
        steps.append(label)

    block = pd.DataFrame({"Step": steps, "Base": base,
                          "Increase": rise, "Decrease": fall})
    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="Waterfall source data", hidden=True)

    chart = BarChart()
    chart.type = "col"
    chart.grouping = "stacked"
    chart.overlap = 100
    chart.title = "Consensus Bridge: Statistical Baseline → Final Plan"
    _style_axes(chart, "", "Units")
    chart.height, chart.width = 10, 22

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + 3,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    if chart.series:
        # The spacer series must be invisible for the waterfall to read.
        chart.series[0].graphicalProperties.noFill = True
        chart.series[0].graphicalProperties.line.noFill = True
    if len(chart.series) > 1:
        chart.series[1].graphicalProperties.solidFill = NAVY
    if len(chart.series) > 2:
        chart.series[2].graphicalProperties.solidFill = RED
    place_chart(ws, chart, anchor)


def add_pareto_chart(ws: Worksheet, df: pd.DataFrame, anchor: str,
                     data_col: int = 40) -> None:
    """
    Root-cause Pareto: value-at-risk bars with a cumulative-percentage line on a
    secondary axis. Bars are ranked by value, not frequency.
    """
    columns = ["Root_Cause", "Value_At_Risk", "Cumulative_Value_Pct"]
    block = df[[c for c in columns if c in df.columns]].copy()
    if block.empty:
        return

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="Pareto source data", hidden=True)

    bars = BarChart()
    bars.type = "col"
    bars.title = "Service Failure Root Causes — Pareto by Value at Risk"
    _style_axes(bars, "Root cause", "Value at risk")
    bars.height, bars.width = 10, 24
    bar_data = Reference(ws, min_col=data_col + 1, min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    bars.add_data(bar_data, titles_from_data=True)
    bars.set_categories(categories)
    if bars.series:
        bars.series[0].graphicalProperties.solidFill = NAVY

    if "Cumulative_Value_Pct" in block.columns:
        line = LineChart()
        line_data = Reference(ws, min_col=data_col + 2, min_row=header, max_row=last)
        line.add_data(line_data, titles_from_data=True)
        line.y_axis.axId = 200
        line.y_axis.title = "Cumulative %"
        line.y_axis.scaling.max = 100
        if line.series:
            _apply_series_colour(line.series[0], AMBER, width=24_000)
            line.series[0].marker = Marker(symbol="circle", size=6)
        bars.y_axis.crosses = "autoZero"
        bars += line

    place_chart(ws, bars, anchor)


def add_inventory_trajectory_chart(ws: Worksheet, df: pd.DataFrame, anchor: str,
                                   data_col: int = 40) -> None:
    """Projected closing inventory against the safety-stock line, by week."""
    columns = ["horizon", "Demand_Units", "Planned_Receipts",
               "Closing_Inventory", "Safety_Stock"]
    block = df[[c for c in columns if c in df.columns]].copy()
    if block.empty:
        return
    block = block.rename(columns={"horizon": "Week"})

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="Inventory trajectory source data", hidden=True)

    chart = LineChart()
    chart.title = "Projected Inventory Trajectory vs Safety Stock"
    _style_axes(chart, "Horizon week", "Units")
    chart.height, chart.width = 10, 24

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + len(block.columns) - 1,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    for index, series in enumerate(chart.series):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        is_safety = index == len(chart.series) - 1
        _apply_series_colour(series, RED if is_safety else colour,
                             width=16_000 if is_safety else 22_000,
                             dashed=is_safety)
    place_chart(ws, chart, anchor)


def add_risk_distribution_chart(ws: Worksheet, counts: Dict[str, int],
                                anchor: str, data_col: int = 46) -> None:
    """RAG risk distribution as a pie chart with value labels."""
    block = pd.DataFrame({"Risk_Band": list(counts.keys()),
                          "SKUs": list(counts.values())})
    if block.empty:
        return
    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title="Risk distribution source data", hidden=True)

    chart = PieChart()
    chart.title = "Supply Risk Distribution"
    chart.height, chart.width = 9, 12
    data = Reference(ws, min_col=data_col + 1, min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showVal = True
    chart.dataLabels.showPercent = True
    place_chart(ws, chart, anchor)


def add_grouped_bar_chart(ws: Worksheet, df: pd.DataFrame, category_col: str,
                          value_cols: Sequence[str], title: str, anchor: str,
                          y_title: str = "", data_col: int = 40,
                          horizontal: bool = False) -> None:
    """General-purpose grouped bar chart used for scenario and model comparisons."""
    keep = [category_col] + [c for c in value_cols if c in df.columns]
    block = df[keep].copy()
    if block.empty or len(keep) < 2:
        return

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title=f"{title} source data", hidden=True)

    chart = BarChart()
    chart.type = "bar" if horizontal else "col"
    chart.title = title
    _style_axes(chart, "", y_title)
    chart.height, chart.width = 9, 22

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + len(keep) - 1,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    for index, series in enumerate(chart.series):
        series.graphicalProperties.solidFill = SERIES_COLOURS[index % len(SERIES_COLOURS)]
    place_chart(ws, chart, anchor)


def add_scatter_chart(ws: Worksheet, df: pd.DataFrame, x_col: str, y_col: str,
                      title: str, anchor: str, data_col: int = 52,
                      max_points: int = 250) -> None:
    """Scatter of two numeric columns — used for cover-weeks versus risk index."""
    block = df[[x_col, y_col]].dropna().head(max_points).copy()
    if block.empty:
        return
    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col,
        title=f"{title} source data", hidden=True)

    chart = ScatterChart()
    chart.title = title
    chart.style = 13
    _style_axes(chart, x_col.replace("_", " "), y_col.replace("_", " "))
    chart.height, chart.width = 10, 18

    x_ref = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    y_ref = Reference(ws, min_col=data_col + 1, min_row=header, max_row=last)
    series = Series(y_ref, x_ref, title_from_data=True)
    series.marker = Marker(symbol="circle", size=5)
    series.graphicalProperties.line.noFill = True
    chart.series.append(series)
    place_chart(ws, chart, anchor)


# --------------------------------------------------------------------------- #
# Advanced worksheet features
# --------------------------------------------------------------------------- #
def add_autofilter(ws: Worksheet, header_row: int, last_row: int,
                   first_col: int, last_col: int) -> None:
    """Enable Excel's native filter dropdowns over a table range."""
    if last_row <= header_row:
        return
    ws.auto_filter.ref = (f"{get_column_letter(first_col)}{header_row}:"
                          f"{get_column_letter(last_col)}{last_row}")


def add_data_bars(ws: Worksheet, cell_range: str, colour: str = BLUE) -> None:
    """
    In-cell data bars.

    On a 500-row table these carry magnitude far faster than reading digits —
    the eye finds the outlier before the brain parses the number.
    """
    ws.conditional_formatting.add(cell_range, DataBarRule(
        start_type="min", end_type="max", color=colour,
        showValue=True, minLength=None, maxLength=None))


def add_icon_set(ws: Worksheet, cell_range: str,
                 icon_style: str = "3TrafficLights1",
                 thresholds: Sequence[float] = (0, 33, 67)) -> None:
    """Traffic-light icon set for RAG-style numeric columns."""
    ws.conditional_formatting.add(cell_range, IconSetRule(
        icon_style, "percent", [str(t) for t in thresholds], showValue=True))


def add_scenario_selector(ws: Worksheet, row: int, col: int,
                          scenarios: Sequence[str],
                          default: str) -> str:
    """
    A real data-validation dropdown, returning the driving cell reference.

    Paired with the formula block below, this makes the workbook genuinely
    interactive: an executive changes one dropdown and the summary figures
    recalculate in Excel, with no Python involved. That is what separates a
    delivered model from an exported report.
    """
    cell = ws.cell(row=row, column=col, value=default)
    cell.font = Font(bold=True, size=11, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="center")

    validation = DataValidation(
        type="list", formula1=f'"{",".join(scenarios)}"', allow_blank=False,
        showDropDown=False, promptTitle="Planning scenario",
        prompt="Select Conservative, Balanced or Aggressive. "
               "The figures below recalculate automatically.")
    ws.add_data_validation(validation)
    validation.add(cell)

    label = ws.cell(row=row, column=col - 1, value="Scenario:")
    label.font = Font(bold=True, size=10)
    label.alignment = Alignment(horizontal="right")
    return f"{get_column_letter(col)}{row}"


def add_lookup_row(ws: Worksheet, row: int, col: int, label: str,
                   selector_ref: str, table_sheet: str, key_range: str,
                   value_range: str, number_format: str = "#,##0") -> None:
    """
    Write a live INDEX/MATCH formula driven by the scenario selector.

    The value is not written by Python — Excel computes it when the workbook is
    opened and recomputes it when the dropdown changes.
    """
    ws.cell(row=row, column=col, value=label).font = Font(size=10, bold=True)
    formula = (f"=IFERROR(INDEX('{table_sheet}'!{value_range},"
               f"MATCH(${selector_ref},'{table_sheet}'!{key_range},0)),\"n/a\")")
    cell = ws.cell(row=row, column=col + 1, value=formula)
    cell.number_format = number_format
    cell.font = Font(size=11, bold=True, color=NAVY)


def setup_print(ws: Worksheet, landscape: bool = True, fit_width: bool = True,
                repeat_header_row: Optional[int] = None,
                title: Optional[str] = None) -> None:
    """
    Print setup, because these workbooks get printed for the S&OP meeting.

    A board pack that spills a 12-column table across three pages is a board
    pack nobody reads.
    """
    ws.page_setup.orientation = "landscape" if landscape else "portrait"
    if fit_width:
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
    if repeat_header_row:
        ws.print_title_rows = f"{repeat_header_row}:{repeat_header_row}"
    ws.oddFooter.center.text = (title or "S&OP Pack") + "  |  Page &P of &N"
    ws.oddFooter.center.size = 8
    ws.oddFooter.center.color = "808080"


def internal_link(ws: Worksheet, row: int, col: int, sheet_name: str,
                  label: Optional[str] = None, size: int = 10) -> None:
    """
    Write a working internal hyperlink to another sheet.

    Assigning a string to ``cell.hyperlink`` makes openpyxl create an EXTERNAL
    relationship, which Excel either ignores or reports as broken. An internal
    jump must be expressed as ``Hyperlink(location=...)`` with no target, so it
    is written as ``<hyperlink location="'Sheet'!A1"/>`` and resolves inside
    the workbook.
    """
    cell = ws.cell(row=row, column=col, value=label or sheet_name)
    cell.hyperlink = Hyperlink(ref=cell.coordinate,
                               location=f"'{sheet_name}'!A1",
                               tooltip=f"Go to {sheet_name}")
    cell.font = Font(size=size, color=BLUE, underline="single")


def add_back_link(ws: Worksheet, row: int = 1, col: int = 15,
                  target: str = "00 Navigation") -> None:
    """
    A persistent 'back to contents' link so no sheet is a dead end.

    Placed at column 15, clear of the title block, which merges columns 1..12.
    Writing into a merged range raises because only the top-left cell of a
    merge is writable.
    """
    if ws.cell(row=row, column=col).__class__.__name__ == "MergedCell":
        col += 4
    internal_link(ws, row, col, target, label="<< Back to Navigation", size=9)


def add_navigation(ws: Worksheet, row: int, sheets: Sequence[str],
                   col: int = 1) -> int:
    """Inline row of internal hyperlinks across a set of sheets."""
    ws.cell(row=row, column=col, value="Jump to:").font = Font(bold=True, size=9,
                                                               color=GREY)
    for offset, sheet in enumerate(sheets, start=1):
        internal_link(ws, row, col + offset, sheet,
                      label=sheet.split(". ")[-1], size=9)
    return row + 2


def add_line_chart(ws: Worksheet, df: pd.DataFrame, category_col: str,
                   value_cols: Sequence[str], title: str, anchor: str,
                   y_title: str = "", data_col: int = 60,
                   dashed_last: bool = False, height: float = 8.5,
                   width: float = 16.0) -> None:
    """
    General-purpose multi-series line chart.

    ``dashed_last`` renders the final series as a dashed reference line, which
    is how targets, safety-stock levels and control limits should read against
    the measured series they govern.
    """
    keep = [category_col] + [c for c in value_cols if c in df.columns]
    block = df[keep].copy()
    if block.empty or len(keep) < 2:
        return

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col, title=f"{title} data", hidden=True)

    chart = LineChart()
    chart.title = title
    _style_axes(chart, "", y_title)
    chart.height, chart.width = height, width

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + len(keep) - 1,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    for index, series in enumerate(chart.series):
        is_last = dashed_last and index == len(chart.series) - 1
        _apply_series_colour(series,
                             RED if is_last else SERIES_COLOURS[index % len(SERIES_COLOURS)],
                             width=16_000 if is_last else 24_000, dashed=is_last)
    place_chart(ws, chart, anchor)


def add_area_band_chart(ws: Worksheet, df: pd.DataFrame, category_col: str,
                        lower_col: str, upper_col: str, value_col: str,
                        title: str, anchor: str, y_title: str = "",
                        data_col: int = 66) -> None:
    """
    Forecast with a prediction band, drawn as three lines.

    Excel has no native banded-interval chart type that openpyxl exposes, so the
    interval is shown as two dashed bounds around a solid central forecast.
    Honest and readable, without pretending to a fill that would not survive a
    round-trip through Excel.
    """
    keep = [category_col, lower_col, value_col, upper_col]
    block = df[[c for c in keep if c in df.columns]].copy()
    if block.empty:
        return

    header, first, last, _ = write_chart_data(
        ws, block, start_row=2, start_col=data_col, title=f"{title} data", hidden=True)

    chart = LineChart()
    chart.title = title
    _style_axes(chart, "", y_title)
    chart.height, chart.width = 8.5, 16.0

    data = Reference(ws, min_col=data_col + 1, max_col=data_col + len(block.columns) - 1,
                     min_row=header, max_row=last)
    categories = Reference(ws, min_col=data_col, min_row=first, max_row=last)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    styles = [(BLUE, True, 12_000), (NAVY, False, 28_000), (BLUE, True, 12_000)]
    for series, (colour, dashed, weight) in zip(chart.series, styles):
        _apply_series_colour(series, colour, width=weight, dashed=dashed)
    place_chart(ws, chart, anchor)
