"""
Module 5 -- Automated S&OP Executive Pack.

Generates the six-sheet Excel workbook that a monthly S&OP board would work
from, plus the supporting analytical workbooks.

Every number in every sheet is read from a pipeline result frame. There are no
typed-in KPIs, no hard-coded totals and no placeholder rows. If an analysis
produced nothing, the sheet says so explicitly rather than showing a plausible
fabricated figure -- an empty finding is still a finding, and a fabricated one
is worse than useless.

Formatting follows consulting-deck conventions because the audience is an
executive committee, not an analyst: a navy header band, banded rows, Indian
lakh/crore number formats, RAG conditional formatting on the columns where a
threshold actually exists, and frozen panes so a 500-row table stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from src.planning import excel_charts as xc
from src.planning.config import AppConfig, Keys, OUTPUT_DIR, ReportConfig
from src.shared.utils import ensure_directories, get_logger, human_number, safe_div

logger = get_logger("saop_pack")

THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


class ExcelStyler:
    """Reusable openpyxl formatting helpers with a single visual language."""

    def __init__(self, cfg: ReportConfig) -> None:
        self.cfg = cfg
        self.header_fill = PatternFill("solid", fgColor=cfg.brand_navy)
        self.subheader_fill = PatternFill("solid", fgColor=cfg.brand_blue)
        self.band_fill = PatternFill("solid", fgColor=cfg.brand_light)
        self.grey_fill = PatternFill("solid", fgColor=cfg.brand_grey)
        self.header_font = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
        self.title_font = Font(bold=True, color=cfg.brand_navy, size=16, name="Calibri")
        self.subtitle_font = Font(italic=True, color="595959", size=9, name="Calibri")
        self.body_font = Font(size=10, name="Calibri")
        self.bold_font = Font(bold=True, size=10, name="Calibri")

    # -- layout ---------------------------------------------------------- #
    def title_block(self, ws: Worksheet, title: str, subtitle: str,
                    width: int = 8, row: int = 1) -> int:
        """Write the standard title band; returns the next free row."""
        ws.cell(row=row, column=1, value=title).font = self.title_font
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(width, 3))
        ws.cell(row=row + 1, column=1, value=subtitle).font = self.subtitle_font
        ws.merge_cells(start_row=row + 1, start_column=1,
                       end_row=row + 1, end_column=max(width, 3))
        stamp = (f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                 f"| {self.cfg.subtitle}")
        ws.cell(row=row + 2, column=1, value=stamp).font = self.subtitle_font
        ws.merge_cells(start_row=row + 2, start_column=1,
                       end_row=row + 2, end_column=max(width, 3))
        ws.row_dimensions[row].height = 22
        return row + 4

    def section(self, ws: Worksheet, row: int, text: str, width: int = 8) -> int:
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = Font(bold=True, color="FFFFFF", size=11, name="Calibri")
        cell.fill = self.subheader_fill
        for col in range(1, max(width, 3) + 1):
            ws.cell(row=row, column=col).fill = self.subheader_fill
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max(width, 3))
        return row + 1

    def write_table(self, ws: Worksheet, df: pd.DataFrame, start_row: int,
                    start_col: int = 1, number_formats: Optional[Dict[str, str]] = None,
                    band: bool = True, max_rows: Optional[int] = None,
                    autofit: bool = True) -> int:
        """
        Write a dataframe as a formatted table; returns the row after the table.

        Large frames are truncated with an explicit note rather than silently
        cut -- a reader must never be left believing they are seeing everything.
        """
        if df is None or df.empty:
            ws.cell(row=start_row, column=start_col,
                    value="No records met the criteria for this analysis.").font = \
                Font(italic=True, color="808080", size=10)
            return start_row + 2

        truncated = False
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
            truncated = True

        number_formats = number_formats or {}

        for offset, column in enumerate(df.columns):
            cell = ws.cell(row=start_row, column=start_col + offset,
                           value=str(column).replace("_", " "))
            cell.font = self.header_font
            cell.fill = self.header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border = BORDER

        for r, (_, record) in enumerate(df.iterrows(), start=1):
            for c, column in enumerate(df.columns):
                value = record[column]
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = None if not np.isfinite(value) else float(value)
                elif isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                elif isinstance(value, pd.Timestamp):
                    value = value.to_pydatetime()
                elif value is not None and not isinstance(value, (int, float, str)):
                    value = str(value)

                cell = ws.cell(row=start_row + r, column=start_col + c, value=value)
                cell.font = self.body_font
                cell.border = BORDER
                if band and r % 2 == 0:
                    cell.fill = self.band_fill
                if column in number_formats:
                    cell.number_format = number_formats[column]
                elif isinstance(value, float):
                    cell.number_format = "#,##0.00"
                elif isinstance(value, int):
                    cell.number_format = "#,##0"

        end_row = start_row + len(df)
        if autofit:
            self.autofit(ws, df, start_col)
        ws.freeze_panes = ws.cell(row=start_row + 1, column=1)

        if truncated:
            note = ws.cell(row=end_row + 1, column=start_col,
                           value=f"Showing top {len(df)} rows. Full dataset available in the "
                                 f"companion CSV export.")
            note.font = Font(italic=True, color="808080", size=9)
            end_row += 1
        return end_row + 2

    @staticmethod
    def autofit(ws: Worksheet, df: pd.DataFrame, start_col: int = 1,
                min_width: int = 10, max_width: int = 42) -> None:
        for offset, column in enumerate(df.columns):
            header_len = len(str(column))
            try:
                sample = df[column].astype(str).head(200)
                body_len = int(sample.str.len().max()) if len(sample) else 0
            except Exception:
                body_len = 0
            width = min(max(header_len + 2, body_len + 2, min_width), max_width)
            ws.column_dimensions[get_column_letter(start_col + offset)].width = width

    # -- KPI cards -------------------------------------------------------- #
    def kpi_row(self, ws: Worksheet, row: int, cards: Sequence[Tuple[str, str, str]]
                ) -> int:
        """
        Render a row of KPI cards: (label, value, RAG status).

        Status drives the colour so the executive eye lands on the exception
        without reading a single number.
        """
        colours = {"good": self.cfg.brand_green, "warn": self.cfg.brand_amber,
                   "bad": self.cfg.brand_red, "neutral": self.cfg.brand_navy}
        for index, (label, value, status) in enumerate(cards):
            col = 1 + index * 2
            label_cell = ws.cell(row=row, column=col, value=label)
            label_cell.font = Font(bold=True, size=9, color="FFFFFF")
            label_cell.fill = PatternFill("solid",
                                          fgColor=colours.get(status, self.cfg.brand_navy))
            label_cell.alignment = Alignment(horizontal="center", wrap_text=True)
            label_cell.border = BORDER
            ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + 1)

            value_cell = ws.cell(row=row + 1, column=col, value=value)
            value_cell.font = Font(bold=True, size=13,
                                   color=colours.get(status, self.cfg.brand_navy))
            value_cell.alignment = Alignment(horizontal="center")
            value_cell.border = BORDER
            ws.merge_cells(start_row=row + 1, start_column=col,
                           end_row=row + 1, end_column=col + 1)
            ws.column_dimensions[get_column_letter(col)].width = 16
            ws.column_dimensions[get_column_letter(col + 1)].width = 16
        ws.row_dimensions[row].height = 28
        ws.row_dimensions[row + 1].height = 22
        return row + 3

    # -- conditional formatting ------------------------------------------- #
    def rag_rule(self, ws: Worksheet, cell_range: str, good_above: float,
                 warn_above: float) -> None:
        """Green above `good_above`, amber above `warn_above`, red below."""
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="greaterThanOrEqual", formula=[str(good_above)],
            fill=PatternFill("solid", fgColor="C6EFCE"), font=Font(color="006100")))
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="greaterThanOrEqual", formula=[str(warn_above)],
            fill=PatternFill("solid", fgColor="FFEB9C"), font=Font(color="9C5700")))
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="lessThan", formula=[str(warn_above)],
            fill=PatternFill("solid", fgColor="FFC7CE"), font=Font(color="9C0006")))

    def reverse_rag_rule(self, ws: Worksheet, cell_range: str, bad_above: float,
                         warn_above: float) -> None:
        """For metrics where high is bad (risk index, utilisation, error)."""
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="greaterThanOrEqual", formula=[str(bad_above)],
            fill=PatternFill("solid", fgColor="FFC7CE"), font=Font(color="9C0006")))
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="greaterThanOrEqual", formula=[str(warn_above)],
            fill=PatternFill("solid", fgColor="FFEB9C"), font=Font(color="9C5700")))
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="lessThan", formula=[str(warn_above)],
            fill=PatternFill("solid", fgColor="C6EFCE"), font=Font(color="006100")))

    def text_highlight(self, ws: Worksheet, cell_range: str,
                       mapping: Dict[str, Tuple[str, str]]) -> None:
        """Highlight rows whose cell text matches a keyword (e.g. Red / BREACH)."""
        for text, (fill_colour, font_colour) in mapping.items():
            ws.conditional_formatting.add(cell_range, FormulaRule(
                formula=[f'EXACT(INDIRECT(ADDRESS(ROW(),COLUMN())),"{text}")'],
                fill=PatternFill("solid", fgColor=fill_colour),
                font=Font(color=font_colour, bold=True)))


class SOPPackBuilder:
    """Builds the executive S&OP workbook and companion analytical workbooks."""

    CURRENCY_FMT = '#,##0'
    PCT_FMT = '0.0"%"'

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: ReportConfig = config.report
        self.styler = ExcelStyler(self.cfg)
        self.currency = self.cfg.currency_symbol
        ensure_directories()

    # ------------------------------------------------------------------ #
    # Sheet 0 -- Cover & navigation
    # ------------------------------------------------------------------ #
    def _cover(self, ws: Worksheet, ctx: Dict[str, object],
               sheet_names: List[str]) -> None:
        s = self.styler
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 62

        ws["A1"] = "S&OP INTELLIGENCE SYSTEM"
        ws["A1"].font = Font(bold=True, size=22, color=self.cfg.brand_navy)
        ws.merge_cells("A1:D1")
        ws["A2"] = "Monthly Sales & Operations Planning Board Pack"
        ws["A2"].font = Font(size=12, color="595959")
        ws.merge_cells("A2:D2")

        row = xc.add_navigation(ws, 4, sheet_names)

        fc, cons, sup, kpi = (ctx["forecast_diag"], ctx["consensus_diag"],
                              ctx["supply_diag"], ctx["kpi_diag"])
        meta = pd.DataFrame([
            {"Item": "Planning scenario", "Detail": ctx["scenario"]},
            {"Item": "Planning horizon", "Detail": f"{ctx['horizon']} weeks"},
            {"Item": "SKUs planned", "Detail": f"{sup['skus_planned']:,}"},
            {"Item": "Forecast models", "Detail": "Holt-Winters, SARIMA, Prophet, "
                                                  "XGBoost, inverse-MAPE ensemble"},
            {"Item": "Validation", "Detail": f"{fc['cv_folds']} rolling-origin CV folds "
                                             f"+ untouched {ctx['horizon']}-week holdout"},
            {"Item": "Orders analysed (OTIF)", "Detail": f"{kpi['total_orders']:,}"},
            {"Item": "Generated", "Detail": datetime.now(timezone.utc).strftime(
                "%d %b %Y %H:%M UTC")},
            {"Item": "Data provenance", "Detail": "SYNTHETIC / SIMULATED for portfolio "
                                                  "demonstration. Not employer or client data."},
        ])
        row = s.section(ws, row, "RUN METADATA", width=4)
        row = s.write_table(ws, meta, row, autofit=False)

        # -- live scenario selector -------------------------------------- #
        row = s.section(ws, row, "INTERACTIVE SCENARIO SELECTOR", width=4)
        note = ws.cell(row=row, column=1,
                       value="Change the dropdown; the figures below recalculate in "
                             "Excel via INDEX/MATCH against the Scenario Data sheet.")
        note.font = Font(italic=True, size=9, color="595959")
        row += 2

        scenarios = list(ctx["scenario_summary"]["Scenario"]) if not ctx[
            "scenario_summary"].empty else ["Balanced"]
        selector = xc.add_scenario_selector(ws, row, 2, scenarios, ctx["scenario"])
        row += 2

        summary = ctx["scenario_summary"]
        if not summary.empty:
            n = len(summary)
            key_range = f"$A$2:$A${n + 1}"
            columns = list(summary.columns)
            for label, column, fmt in [
                ("Consensus units", "Consensus_Units", "#,##0"),
                ("Consensus value", "Consensus_Value",
                 f'"{self.currency}"#,##0'),
                ("Constrained-out units", "Constrained_Out_Units", "#,##0"),
                ("Constrained-out value", "Constrained_Out_Value",
                 f'"{self.currency}"#,##0'),
                ("Override alerts", "Override_Alerts", "#,##0"),
                ("Constrained SKUs", "Constrained_SKUs", "#,##0"),
            ]:
                if column not in columns:
                    continue
                letter = get_column_letter(columns.index(column) + 1)
                xc.add_lookup_row(ws, row, 1, label, selector, "Scenario Data",
                                  key_range, f"${letter}$2:${letter}${n + 1}", fmt)
                row += 1

        row += 1
        row = s.section(ws, row, "HOW TO READ THIS PACK", width=4)
        guide = pd.DataFrame([
            {"Sheet": "1. Executive Summary", "Purpose": "Headline position, accuracy "
             "ladder, service, decisions required"},
            {"Sheet": "2. Demand Review", "Purpose": "Model competition, category bias, "
             "12-week plan, poorly forecastable SKUs"},
            {"Sheet": "3. Supply Review", "Purpose": "Capacity breaches, shortages, "
             "risk register, recovery actions"},
            {"Sheet": "4. Inventory Review", "Purpose": "Trajectory, cover, safety-stock "
             "policy, excess exposure"},
            {"Sheet": "5. Financial Recon", "Purpose": "Demand-to-revenue bridge, "
             "scenario impact, working capital"},
            {"Sheet": "6. Decisions Required", "Purpose": "Auto-generated action register "
             "with owners"},
            {"Sheet": "Scenario Data", "Purpose": "Lookup table driving the selector above"},
        ])
        s.write_table(ws, guide, row, autofit=False)
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, landscape=False, title="S&OP Pack — Cover")

    # ------------------------------------------------------------------ #
    # Sheet 1 -- Executive Summary
    # ------------------------------------------------------------------ #
    def _executive_summary(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "S&OP EXECUTIVE SUMMARY",
                            f"Monthly Sales & Operations Planning Review  |  "
                            f"{ctx['scenario']} Scenario  |  "
                            f"{ctx['horizon']}-Week Planning Horizon", width=12)

        fc, cons, sup, kpi = (ctx["forecast_diag"], ctx["consensus_diag"],
                              ctx["supply_diag"], ctx["kpi_diag"])

        otif = kpi["overall_otif_pct"]
        otif_status = ("good" if otif >= self.config.kpi.otif_world_class
                       else "warn" if otif >= self.config.kpi.otif_acceptable else "bad")
        accuracy = ctx.get("review_level_wape")
        acc_status = ("good" if accuracy is not None
                      and accuracy < self.config.forecast.portfolio_mape_target else "warn")
        risk_counts = sup.get("risk_distribution", {})
        red = int(risk_counts.get("Red", 0))
        risk_status = "bad" if red > 20 else "warn" if red else "good"

        row = s.section(ws, row, "HEADLINE POSITION", width=12)
        row = s.kpi_row(ws, row, [
            ("Consensus Demand", human_number(cons["consensus_value"], self.currency), "neutral"),
            ("Forecast Accuracy\n(Cat/Month WAPE)",
             f"{accuracy:.1f}%" if accuracy is not None else "n/a", acc_status),
            ("OTIF Service", f"{otif:.1f}%", otif_status),
            ("Red-Risk SKUs", f"{red:,}", risk_status),
        ])
        row = s.kpi_row(ws, row, [
            ("Working Capital", human_number(sup["total_working_capital"], self.currency),
             "neutral"),
            ("Revenue At Risk", human_number(sup.get("revenue_at_risk", 0), self.currency),
             "bad" if sup.get("revenue_at_risk", 0) > 0 else "good"),
            ("Capacity Breaches", f"{sup['capacity_breaches']:,}",
             "bad" if sup["capacity_breaches"] else "good"),
            ("Override Alerts", f"{cons['override_alerts']:,}",
             "warn" if cons["override_alerts"] else "good"),
        ])

        row = s.section(ws, row, "DEMAND OUTLOOK", width=12)
        demand_table = pd.DataFrame([
            {"Measure": "Statistical baseline (units)",
             "Value": round(ctx["statistical_units"], 0),
             "Commentary": "Model output before commercial input"},
            {"Measure": "Consensus plan (units)",
             "Value": round(cons["consensus_units"], 0),
             "Commentary": f"{ctx['scenario']} scenario, post-constraint"},
            {"Measure": "Consensus plan (value)",
             "Value": round(cons["consensus_value"], 0),
             "Commentary": f"At list price, {self.currency}"},
            {"Measure": "Demand constrained out (units)",
             "Value": round(cons["constrained_out_units"], 0),
             "Commentary": "Cannot be supplied within capacity"},
            {"Measure": "Demand constrained out (value)",
             "Value": round(cons["constrained_out_value"], 0),
             "Commentary": "Revenue forgone unless capacity secured"},
            {"Measure": "SKUs with commercial override",
             "Value": cons["skus_with_commercial_input"],
             "Commentary": "Sales and marketing input applied"},
        ])
        row = s.write_table(ws, demand_table, row,
                            number_formats={"Value": self.CURRENCY_FMT})

        row = s.section(ws, row, "FORECAST ACCURACY BY AGGREGATION LEVEL", width=12)
        hierarchy = ctx.get("hierarchy")
        if hierarchy is not None and not hierarchy.empty:
            start = row
            row = s.write_table(
                ws, hierarchy, row,
                number_formats={"MAPE": "0.00", "WAPE": "0.00", "Bias_Pct": "+0.00;-0.00"})
            wape_col = list(hierarchy.columns).index("WAPE") + 1
            s.reverse_rag_rule(
                ws,
                f"{get_column_letter(wape_col)}{start + 1}:"
                f"{get_column_letter(wape_col)}{start + len(hierarchy)}",
                bad_above=30, warn_above=15)
            note = ws.cell(row=row, column=1,
                           value="Accuracy is measured on an untouched holdout window. Error "
                                 "cancels as demand is aggregated, which is why the review "
                                 "level and the SKU-week level differ so sharply.")
            note.font = Font(italic=True, size=9, color="595959")
            row += 2
            xc.add_accuracy_ladder_chart(
                ws, hierarchy, f"A{row}", data_col=40,
                benchmark=self.config.forecast.portfolio_mape_target)
            row += 19

        spc = ctx.get("spc_chart")
        if spc is not None and not spc.empty:
            row = s.section(ws, row, "SERVICE CONTROL CHART", width=12)
            xc.add_spc_control_chart(ws, spc, f"A{row}", data_col=46)
            row += 23

        row = s.section(ws, row, "SERVICE PERFORMANCE", width=12)
        service_table = pd.DataFrame([
            {"Measure": "OTIF", "Value": otif,
             "Benchmark": kpi["benchmark"]},
            {"Measure": "On Time", "Value": kpi["on_time_pct"], "Benchmark": "-"},
            {"Measure": "In Full", "Value": kpi["in_full_pct"], "Benchmark": "-"},
            {"Measure": "Volume Fill Rate", "Value": kpi["volume_fill_rate_pct"],
             "Benchmark": "-"},
            {"Measure": "Special-cause months (SPC)",
             "Value": kpi["special_cause_months"], "Benchmark": "Target: 0"},
        ])
        row = s.write_table(ws, service_table, row, number_formats={"Value": "0.00"})

        row = s.section(ws, row, "KEY DECISIONS REQUIRED", width=12)
        insights = ctx.get("insights")
        if insights is not None and not insights.empty:
            top = insights[insights["Severity"].isin(["Critical", "High"])].head(8)
            if top.empty:
                top = insights.head(6)
            decisions = top[["Severity", "Domain", "Headline", "Owner",
                             "Recommended_Action"]].copy()
            start = row
            row = s.write_table(ws, decisions, row)
            s.text_highlight(
                ws,
                f"A{start + 1}:A{start + len(decisions)}",
                {"Critical": ("FFC7CE", "9C0006"), "High": ("FFEB9C", "9C5700")})
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Sheet 2 -- Demand Review
    # ------------------------------------------------------------------ #
    def _demand_review(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "DEMAND REVIEW",
                            "Statistical baseline, model performance and demand risk",
                            width=12)

        row = s.section(ws, row, "MODEL LEADERBOARD (HOLDOUT)", width=12)
        leaderboard = ctx.get("leaderboard")
        if leaderboard is not None and not leaderboard.empty:
            holdout = leaderboard[leaderboard["Stage"] == "HOLDOUT"]
            row = s.write_table(ws, holdout if not holdout.empty else leaderboard, row,
                                number_formats={"Median_MAPE": "0.00", "Median_WAPE": "0.00"})

        leaderboard_frame = ctx.get("leaderboard")
        if leaderboard_frame is not None and not leaderboard_frame.empty:
            holdout = leaderboard_frame[leaderboard_frame["Stage"] == "HOLDOUT"]
            panel_rows = holdout[holdout["Tier"] == "Competition Panel"]
            if not panel_rows.empty:
                xc.add_grouped_bar_chart(
                    ws, panel_rows, "Model", ["Median_MAPE", "Median_WAPE"],
                    "Model Competition — Untouched Holdout (competition panel)",
                    f"A{row}", y_title="Error (%)", data_col=52)
                row += 19

        row = s.section(ws, row, "CATEGORY ACCURACY AND BIAS", width=12)
        category = ctx.get("category_accuracy")
        if category is not None and not category.empty:
            start = row
            row = s.write_table(ws, category, row,
                                number_formats={"Weighted_Bias_Pct": "+0.00;-0.00"})
            if "Weighted_Bias_Pct" in category.columns:
                col = get_column_letter(list(category.columns).index("Weighted_Bias_Pct") + 1)
                ws.conditional_formatting.add(
                    f"{col}{start + 1}:{col}{start + len(category)}",
                    ColorScaleRule(start_type="num", start_value=-25, start_color="F8696B",
                                   mid_type="num", mid_value=0, mid_color="FFFFFF",
                                   end_type="num", end_value=25, end_color="F8696B"))

        row = s.section(ws, row, "12-WEEK DEMAND PLAN BY CATEGORY AND WEEK", width=12)
        pivot = ctx.get("demand_pivot")
        row = s.write_table(ws, pivot, row)

        row = s.section(ws, row, "HIGH-RISK SKUs - POOR FORECASTABILITY", width=12)
        accuracy = ctx.get("sku_accuracy")
        if accuracy is not None and not accuracy.empty:
            worst = accuracy.sort_values("MAPE", ascending=False).head(30)
            keep = [c for c in [Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, "Selected_Model",
                                "MAPE", "WAPE", "Bias_Pct", "Accuracy_Band", "Total_Actual"]
                    if c in worst.columns]
            start = row
            row = s.write_table(ws, worst[keep], row,
                                number_formats={"MAPE": "0.0", "WAPE": "0.0"})
            if "MAPE" in keep:
                col = get_column_letter(keep.index("MAPE") + 1)
                s.reverse_rag_rule(ws, f"{col}{start + 1}:{col}{start + len(worst)}",
                                   bad_above=50, warn_above=25)

        row = s.section(ws, row, "PROMOTIONAL ACTIVITY IN THE HORIZON", width=12)
        row = s.write_table(ws, ctx.get("promo_summary"), row)
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Sheet 3 -- Supply Review
    # ------------------------------------------------------------------ #
    def _supply_review(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "SUPPLY REVIEW",
                            "Capacity, constraints, shortages and recovery actions", width=12)

        row = s.section(ws, row, "DEMAND VERSUS SUPPLY BY WEEK", width=12)
        row = s.write_table(ws, ctx.get("supply_vs_demand"), row,
                            number_formats={"Demand_Units": self.CURRENCY_FMT,
                                            "Planned_Receipts": self.CURRENCY_FMT})

        supply_vs_demand = ctx.get("supply_vs_demand")
        if supply_vs_demand is not None and not supply_vs_demand.empty:
            xc.add_grouped_bar_chart(
                ws, supply_vs_demand, "horizon",
                ["Demand_Units", "Planned_Receipts", "Open_PO_Receipts"],
                "Demand vs Planned Supply by Week", f"A{row}",
                y_title="Units", data_col=52)
            row += 19

        risk_counts = ctx["supply_diag"].get("risk_distribution", {})
        if risk_counts:
            xc.add_risk_distribution_chart(ws, risk_counts, f"L{row - 19}"
                                           if row > 19 else "L2", data_col=60)

        row = s.section(ws, row, "SUPPLIER CAPACITY CONSTRAINTS", width=12)
        capacity = ctx.get("capacity_alerts")
        if capacity is not None and not capacity.empty:
            flagged = capacity[capacity["Severity"] != "OK"]
            display = flagged if not flagged.empty else capacity.head(20)
            start = row
            row = s.write_table(ws, display, row, max_rows=120,
                                number_formats={"Utilisation_Pct": "0.0"})
            if "Utilisation_Pct" in display.columns:
                col = get_column_letter(list(display.columns).index("Utilisation_Pct") + 1)
                n = min(len(display), 120)
                s.reverse_rag_rule(ws, f"{col}{start + 1}:{col}{start + n}",
                                   bad_above=100, warn_above=95)
            if flagged.empty:
                note = ws.cell(row=row, column=1,
                               value="No capacity breaches or warnings detected in the "
                                     "horizon. Showing the highest-utilisation resources "
                                     "for reference.")
                note.font = Font(italic=True, size=9, color="595959")
                row += 2

        row = s.section(ws, row, "CONSTRAINED SKUs - PROJECTED SHORTAGES", width=12)
        stockouts = ctx.get("stockout_risks")
        if stockouts is not None and not stockouts.empty:
            projected = stockouts[stockouts["Event_Type"] == "Projected Stockout"]
            display = projected if not projected.empty else stockouts
            keep = [c for c in [Keys.SKU, Keys.CATEGORY, Keys.SUPPLIER, "horizon",
                                "Event_Type", "Shortage_Units", "Revenue_At_Risk",
                                "Lead_Time_Weeks", "Can_Recover_In_Lead_Time",
                                "Supply_Risk_Index", "Risk_Band", "Recommended_Action"]
                    if c in display.columns]
            row = s.write_table(ws, display[keep].head(60), row,
                                number_formats={"Revenue_At_Risk": self.CURRENCY_FMT})
        else:
            row = s.write_table(ws, pd.DataFrame(), row)

        row = s.section(ws, row, "SUPPLY RISK INDEX - HIGHEST RISK SKUs", width=12)
        summary = ctx.get("sku_summary")
        if summary is not None and not summary.empty:
            keep = [c for c in [Keys.SKU, Keys.CATEGORY, Keys.SUPPLIER, "ABC_XYZ",
                                "Supply_Risk_Index", "Risk_Band", "Min_Cover",
                                "First_Stockout_Week", "Supplier_Peak_Utilisation_Pct",
                                "Lead_Time_Weeks", "Recommended_Action"]
                    if c in summary.columns]
            top = summary.head(40)[keep]
            start = row
            row = s.write_table(ws, top, row,
                                number_formats={"Supply_Risk_Index": "0.0"})
            if "Supply_Risk_Index" in keep:
                col = get_column_letter(keep.index("Supply_Risk_Index") + 1)
                s.reverse_rag_rule(
                    ws, f"{col}{start + 1}:{col}{start + len(top)}",
                    bad_above=self.config.supply.risk_red_threshold,
                    warn_above=self.config.supply.risk_amber_threshold)

        row = s.section(ws, row, "RECOVERY ACTIONS BY PRIORITY", width=12)
        row = s.write_table(ws, ctx.get("recovery_actions"), row)
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Sheet 4 -- Inventory Review
    # ------------------------------------------------------------------ #
    def _inventory_review(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "INVENTORY REVIEW",
                            "Projected inventory trajectory, cover and excess exposure",
                            width=12)

        row = s.section(ws, row, "PROJECTED INVENTORY TRAJECTORY", width=12)
        row = s.write_table(ws, ctx.get("inventory_trajectory"), row,
                            number_formats={"Opening_Inventory": self.CURRENCY_FMT,
                                            "Closing_Inventory": self.CURRENCY_FMT,
                                            "Weeks_Of_Cover": "0.00"})

        trajectory = ctx.get("inventory_trajectory")
        if trajectory is not None and not trajectory.empty:
            xc.add_inventory_trajectory_chart(ws, trajectory, f"A{row}", data_col=46)
            row += 21

        row = s.section(ws, row, "INVENTORY STATUS DISTRIBUTION", width=12)
        row = s.write_table(ws, ctx.get("inventory_status"), row)

        row = s.section(ws, row, "SAFETY STOCK POLICY BY ABC-XYZ CLASS", width=12)
        row = s.write_table(ws, ctx.get("policy_summary"), row,
                            number_formats={"Safety_Stock_Value": self.CURRENCY_FMT})

        row = s.section(ws, row, "EXCESS AND SLOW-MOVING STOCK", width=12)
        excess = ctx.get("excess_stock")
        if excess is not None and not excess.empty:
            row = s.write_table(ws, excess.head(40), row,
                                number_formats={"Avg_Working_Capital": self.CURRENCY_FMT})
        else:
            row = s.write_table(ws, pd.DataFrame(), row)

        row = s.section(ws, row, "STOCKOUT RISK SUMMARY", width=12)
        row = s.write_table(ws, ctx.get("stockout_summary"), row)
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Sheet 5 -- Financial Reconciliation
    # ------------------------------------------------------------------ #
    def _financial_reconciliation(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "FINANCIAL RECONCILIATION",
                            "Reconciling the demand plan to revenue, inventory and "
                            "working capital", width=12)

        row = s.section(ws, row, "DEMAND TO REVENUE BRIDGE", width=12)
        row = s.write_table(ws, ctx.get("financial_bridge"), row,
                            number_formats={"Value": self.CURRENCY_FMT})

        waterfall = ctx.get("waterfall")
        if waterfall is not None and not waterfall.empty:
            xc.add_waterfall_chart(ws, waterfall, f"A{row}", data_col=46)
            row += 21

        row = s.section(ws, row, "SCENARIO FINANCIAL IMPACT", width=12)
        scenarios = ctx.get("scenario_summary")
        if scenarios is not None and not scenarios.empty:
            row = s.write_table(ws, scenarios, row,
                                number_formats={"Consensus_Value": self.CURRENCY_FMT,
                                                "Constrained_Out_Value": self.CURRENCY_FMT})
            note = ws.cell(row=row, column=1,
                           value="Scenario selection is a working-capital and service "
                                 "trade-off, not a forecasting decision. The spread is the "
                                 "quantified cost of the risk-appetite choice.")
            note.font = Font(italic=True, size=9, color="595959")
            row += 2

        row = s.section(ws, row, "WORKING CAPITAL BY WEEK", width=12)
        row = s.write_table(ws, ctx.get("working_capital_profile"), row,
                            number_formats={"Working_Capital": self.CURRENCY_FMT})

        scenarios_frame = ctx.get("scenario_summary")
        if scenarios_frame is not None and not scenarios_frame.empty:
            xc.add_grouped_bar_chart(
                ws, scenarios_frame, "Scenario",
                ["Consensus_Units", "Constrained_Out_Units"],
                "Scenario Comparison — Achievable vs Constrained-Out Volume",
                f"A{row}", y_title="Units", data_col=52)
            row += 19

        row = s.section(ws, row, "VALUE AT RISK", width=12)
        row = s.write_table(ws, ctx.get("value_at_risk"), row,
                            number_formats={"Value": self.CURRENCY_FMT})
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Sheet 6 -- Decisions Required
    # ------------------------------------------------------------------ #
    def _decisions_required(self, ws: Worksheet, ctx: Dict[str, object]) -> None:
        s = self.styler
        row = s.title_block(ws, "DECISIONS REQUIRED",
                            "Actions generated automatically from calculated thresholds",
                            width=12)

        decisions = ctx.get("decisions")
        row = s.section(ws, row, "ACTION REGISTER", width=12)
        if decisions is not None and not decisions.empty:
            start = row
            row = s.write_table(ws, decisions, row,
                                number_formats={"Impact_Value": self.CURRENCY_FMT})
            s.text_highlight(
                ws, f"B{start + 1}:B{start + len(decisions)}",
                {"Critical": ("FFC7CE", "9C0006"),
                 "High": ("FFEB9C", "9C5700"),
                 "Medium": ("DDEBF7", "1F3864")})
        else:
            row = s.write_table(ws, pd.DataFrame(), row)

        row = s.section(ws, row, "OVERRIDE GOVERNANCE - ITEMS REQUIRING SIGN-OFF", width=12)
        overrides = ctx.get("override_escalations")
        row = s.write_table(ws, overrides, row, max_rows=60,
                            number_formats={"Override_Deviation_Pct": "+0.0;-0.0"})

        row = s.section(ws, row, "METHODOLOGY NOTE", width=12)
        method = pd.DataFrame([
            {"Item": "Data provenance",
             "Statement": "All data is synthetic and simulated for portfolio demonstration. "
                          "It does not originate from any employer or client."},
            {"Item": "Forecast accuracy",
             "Statement": "Measured on a 12-week holdout window that is excluded from model "
                          "selection and ensemble weighting."},
            {"Item": "Action generation",
             "Statement": "Every action in this register is derived from a calculated metric "
                          "crossing a configured threshold. None are hard-coded."},
            {"Item": "Supply Risk Index",
             "Statement": "Weighted blend of stockout proximity, capacity utilisation, cover "
                          "shortfall, lead-time risk and forecast uncertainty. Weights are "
                          "documented in METHODOLOGY.md."},
            {"Item": "SPC limits",
             "Statement": "Sigma estimated from the average moving range (MRbar / 1.128), "
                          "not the raw standard deviation."},
        ])
        row = s.write_table(ws, method, row)
        ws.sheet_view.showGridLines = False

    # ------------------------------------------------------------------ #
    # Context assembly
    # ------------------------------------------------------------------ #
    def _build_context(self, forecast_result, consensus_result, supply_result,
                       kpi_result, insights: pd.DataFrame) -> Dict[str, object]:
        """Derive every table the workbook needs from the pipeline results."""
        scenario = consensus_result.diagnostics["scenario"]
        plan = consensus_result.consensus_plan
        primary = plan[plan["Scenario"] == scenario]
        supply = supply_result.supply_plan
        summary = supply_result.sku_summary
        policy = supply_result.inventory_policy

        hierarchy = forecast_result.hierarchy_accuracy
        review_wape = None
        if not hierarchy.empty:
            match = hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
            if not match.empty:
                review_wape = float(match.iloc[0]["WAPE"])

        # -- demand pivot ------------------------------------------------- #
        demand_pivot = (primary.pivot_table(index=Keys.CATEGORY, columns="horizon",
                                            values="Consensus_Units", aggfunc="sum")
                        .round(0).reset_index())
        demand_pivot.columns = [Keys.CATEGORY if c == Keys.CATEGORY else f"W{c}"
                                for c in demand_pivot.columns]
        demand_pivot["Total"] = demand_pivot.drop(columns=[Keys.CATEGORY]).sum(axis=1)

        # -- promotions in horizon ---------------------------------------- #
        promo_summary = (primary.groupby("horizon", as_index=False)
                         .agg(Promo_SKUs=(Keys.SKU, "nunique"),
                              Consensus_Units=("Consensus_Units", "sum"))
                         if "horizon" in primary.columns else pd.DataFrame())
        if "Is_Promo" in primary.columns:
            promo_summary = (primary.groupby("horizon", as_index=False)
                             .agg(SKUs_On_Promotion=("Is_Promo", "sum"),
                                  Consensus_Units=("Consensus_Units", "sum"),
                                  Consensus_Value=("Consensus_Value", "sum")))
            promo_summary["Consensus_Units"] = promo_summary["Consensus_Units"].round(0)
            promo_summary["Consensus_Value"] = promo_summary["Consensus_Value"].round(0)

        # -- supply vs demand --------------------------------------------- #
        supply_vs_demand = (supply.groupby("horizon", as_index=False)
                            .agg(Demand_Units=("Demand_Units", "sum"),
                                 Opening_Inventory=("Opening_Inventory", "sum"),
                                 Planned_Receipts=("Planned_Receipts", "sum"),
                                 Open_PO_Receipts=("Open_PO_Receipts", "sum"),
                                 Planned_Orders=("Planned_Order_Qty", "sum"),
                                 Closing_Inventory=("Closing_Inventory", "sum"),
                                 Shortage_Units=("Shortage_Units", "sum")))
        supply_vs_demand["Supply_Coverage_Pct"] = (
            (supply_vs_demand["Opening_Inventory"] + supply_vs_demand["Planned_Receipts"]
             + supply_vs_demand["Open_PO_Receipts"])
            / supply_vs_demand["Demand_Units"].replace(0, np.nan) * 100).round(1)
        supply_vs_demand = supply_vs_demand.round(0)

        # -- recovery actions --------------------------------------------- #
        recovery = (summary[summary["Recommended_Action"] != "No action - within policy"]
                    .groupby("Recommended_Action", as_index=False)
                    .agg(SKUs_Affected=(Keys.SKU, "nunique"),
                         Avg_Risk_Index=("Supply_Risk_Index", "mean"),
                         Total_Demand=("Total_Demand", "sum")))
        if not recovery.empty:
            recovery["Avg_Risk_Index"] = recovery["Avg_Risk_Index"].round(1)
            recovery["Total_Demand"] = recovery["Total_Demand"].round(0)
            recovery = recovery.sort_values("Avg_Risk_Index", ascending=False)

        # -- inventory ----------------------------------------------------- #
        trajectory = (supply.groupby("horizon", as_index=False)
                      .agg(Opening_Inventory=("Opening_Inventory", "sum"),
                           Demand_Units=("Demand_Units", "sum"),
                           Planned_Receipts=("Planned_Receipts", "sum"),
                           Closing_Inventory=("Closing_Inventory", "sum"),
                           Safety_Stock=("Safety_Stock", "sum"),
                           Weeks_Of_Cover=("Weeks_Of_Cover", "mean"),
                           Working_Capital=("Working_Capital", "sum")).round(2))

        inventory_status = (supply.groupby(["horizon", "Inventory_Status"])
                            .size().unstack(fill_value=0).reset_index())

        policy_summary = (policy.groupby("ABC_XYZ", as_index=False)
                          .agg(SKUs=(Keys.SKU, "nunique"),
                               Avg_Service_Level=("Service_Level_Target", "mean"),
                               Avg_Safety_Stock_Units=("Safety_Stock_Units", "mean"),
                               Safety_Stock_Value=("Safety_Stock_Value", "sum"),
                               Avg_Lead_Time_Weeks=("Lead_Time_Weeks", "mean"),
                               Avg_Cover_Weeks=("Safety_Stock_Weeks", "mean")).round(2))

        excess = summary[summary["Min_Cover"] > self.config.supply.excess_cover_weeks]
        excess_cols = [c for c in [Keys.SKU, Keys.CATEGORY, "ABC_Class", "Min_Cover",
                                   "Avg_Cover", "Avg_Working_Capital", "Total_Demand"]
                       if c in excess.columns]
        excess_display = (excess[excess_cols].sort_values("Avg_Working_Capital",
                                                          ascending=False)
                          if not excess.empty else pd.DataFrame())

        stockout_summary = pd.DataFrame([
            {"Measure": "SKUs with projected stockout",
             "Value": supply_result.diagnostics["skus_with_stockout"]},
            {"Measure": "SKUs dipping below safety stock",
             "Value": supply_result.diagnostics["skus_below_safety_stock"]},
            {"Measure": "SKUs requiring expedite (inside lead time)",
             "Value": supply_result.diagnostics["expedite_required"]},
            {"Measure": "Earliest projected stockout (week)",
             "Value": supply_result.diagnostics.get("earliest_stockout_week") or 0},
            {"Measure": "Average weeks of cover",
             "Value": round(supply_result.diagnostics["avg_weeks_of_cover"], 2)},
        ])

        # -- financials ---------------------------------------------------- #
        statistical_units = float(primary["Forecast_Units"].sum())
        statistical_value = float((primary["Forecast_Units"]
                                   * primary["Unit_Price"]).sum())
        financial_bridge = pd.DataFrame([
            {"Line": "Statistical demand value", "Value": round(statistical_value, 0)},
            {"Line": "Commercial adjustment",
             "Value": round(float((primary["Commercial_Forecast"]
                                   - primary["Forecast_Units"]).mul(
                 primary["Unit_Price"]).sum()), 0)},
            {"Line": "Constrained out",
             "Value": -round(float(primary["Constrained_Out_Value"].sum()), 0)},
            {"Line": "Consensus plan value",
             "Value": round(float(primary["Consensus_Value"].sum()), 0)},
            {"Line": "Inventory value at week 1",
             "Value": round(float(supply.loc[supply["horizon"] == 1,
                                             "Working_Capital"].sum()), 0)},
            {"Line": "Safety stock value",
             "Value": round(float(policy["Safety_Stock_Value"].sum()), 0)},
        ])

        working_capital_profile = (supply.groupby("horizon", as_index=False)
                                   .agg(Working_Capital=("Working_Capital", "sum"),
                                        Closing_Inventory=("Closing_Inventory", "sum"))
                                   .round(0))

        value_at_risk = pd.DataFrame([
            {"Exposure": "Revenue at risk from projected stockouts",
             "Value": round(supply_result.diagnostics.get("revenue_at_risk", 0), 0)},
            {"Exposure": "Demand constrained out by capacity",
             "Value": round(consensus_result.diagnostics["constrained_out_value"], 0)},
            {"Exposure": "Working capital in excess-cover SKUs",
             "Value": round(float(excess["Avg_Working_Capital"].sum())
                            if not excess.empty else 0, 0)},
            {"Exposure": "Service-failure value (trailing 12 months)",
             "Value": round(float(kpi_result.root_cause_analysis["Value_At_Risk"].sum())
                            if not kpi_result.root_cause_analysis.empty else 0, 0)},
        ])

        # -- decisions ------------------------------------------------------ #
        decisions = pd.DataFrame()
        if insights is not None and not insights.empty:
            cols = [c for c in ["Rank", "Severity", "Domain", "Headline",
                                "Recommended_Action", "Owner", "Metric", "Impact_Value"]
                    if c in insights.columns]
            decisions = insights[cols].copy()

        override_log = consensus_result.override_log
        escalations = pd.DataFrame()
        if not override_log.empty and "Requires_Board_Signoff" in override_log.columns:
            flagged = override_log[override_log["Requires_Board_Signoff"]]
            if flagged.empty:
                flagged = override_log[override_log["Override_Alert"]]
            keep = [c for c in [Keys.SKU, Keys.CATEGORY, "horizon", "Statistical_Forecast",
                                "Commercial_Override", "Override_Deviation_Pct",
                                "Commercial_Driver", "Justification", "Submitted_By",
                                "Governance_Status", "Planner_Decision"]
                    if c in flagged.columns]
            escalations = flagged[keep] if not flagged.empty else pd.DataFrame()

        return {
            "scenario": scenario,
            "horizon": self.config.supply.horizon_weeks,
            "forecast_diag": forecast_result.diagnostics,
            "consensus_diag": consensus_result.diagnostics,
            "supply_diag": supply_result.diagnostics,
            "kpi_diag": kpi_result.diagnostics,
            "review_level_wape": review_wape,
            "hierarchy": hierarchy,
            "leaderboard": forecast_result.model_leaderboard,
            "category_accuracy": forecast_result.category_accuracy,
            "sku_accuracy": forecast_result.sku_accuracy,
            "statistical_units": statistical_units,
            "demand_pivot": demand_pivot,
            "promo_summary": promo_summary,
            "supply_vs_demand": supply_vs_demand,
            "capacity_alerts": supply_result.capacity_alerts,
            "stockout_risks": supply_result.stockout_risks,
            "sku_summary": summary,
            "recovery_actions": recovery,
            "inventory_trajectory": trajectory,
            "inventory_status": inventory_status,
            "policy_summary": policy_summary,
            "excess_stock": excess_display,
            "stockout_summary": stockout_summary,
            "financial_bridge": financial_bridge,
            "scenario_summary": consensus_result.scenario_summary,
            "working_capital_profile": working_capital_profile,
            "value_at_risk": value_at_risk,
            "decisions": decisions,
            "override_escalations": escalations,
            "insights": insights,
            "spc_chart": kpi_result.spc_chart,
            "waterfall": consensus_result.waterfall,
            "root_causes": kpi_result.root_cause_analysis,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_sop_pack(self, forecast_result, consensus_result, supply_result,
                       kpi_result, insights: pd.DataFrame) -> Path:
        """Write the six-sheet executive S&OP workbook."""
        ctx = self._build_context(forecast_result, consensus_result, supply_result,
                                  kpi_result, insights)
        wb = Workbook()
        wb.remove(wb.active)

        builders = [
            ("1. Executive Summary", self._executive_summary),
            ("2. Demand Review", self._demand_review),
            ("3. Supply Review", self._supply_review),
            ("4. Inventory Review", self._inventory_review),
            ("5. Financial Recon", self._financial_reconciliation),
            ("6. Decisions Required", self._decisions_required),
        ]
        sheet_names = [name for name, _ in builders]

        cover = wb.create_sheet("Cover")
        for name, builder in builders:
            builder(wb.create_sheet(name), ctx)

        # Lookup table backing the cover sheet's live scenario selector. Written
        # last so the formulas on the cover always resolve against real data.
        scenario_ws = wb.create_sheet("Scenario Data")
        self.styler.write_table(scenario_ws, ctx["scenario_summary"], 1)
        scenario_ws.sheet_view.showGridLines = False

        # Root-cause Pareto lives beside the service analysis it explains.
        causes = ctx.get("root_causes")
        if causes is not None and not causes.empty:
            pareto_ws = wb.create_sheet("Root Cause Pareto")
            row = self.styler.title_block(
                pareto_ws, "ROOT CAUSE PARETO",
                "Service failures ranked by value at risk, not frequency", width=10)
            xc.add_pareto_chart(pareto_ws, causes, f"A{row}", data_col=40)
            row += 22
            self.styler.write_table(pareto_ws, causes, row)
            pareto_ws.sheet_view.showGridLines = False
            xc.setup_print(pareto_ws, title="Root Cause Pareto")

        self._cover(cover, ctx, sheet_names)

        for name in sheet_names:
            xc.setup_print(wb[name], title=name)

        path = OUTPUT_DIR / "sop_pack.xlsx"
        wb.save(path)
        logger.info("S&OP executive pack written: %s (%d sheets, native charts).",
                    path.name, len(wb.worksheets))
        return path

    def build_supporting_workbooks(self, forecast_result, consensus_result,
                                   supply_result, kpi_result,
                                   insights: pd.DataFrame) -> Dict[str, Path]:
        """
        Write the analytical workbooks that sit behind the executive pack.

        These carry the full detail an analyst needs to interrogate any number
        in the board pack -- the pack is the summary, these are the evidence.
        """
        s = self.styler
        outputs: Dict[str, Path] = {}

        specs: List[Tuple[str, str, List[Tuple[str, pd.DataFrame]]]] = [
            ("demand_forecast.xlsx", "12-Week Demand Forecast", [
                ("Forecast", forecast_result.forecast),
                ("Uncertainty (Output 5 Feed)", forecast_result.uncertainty),
                ("Ensemble Weights", forecast_result.ensemble_weights),
            ]),
            ("forecast_accuracy.xlsx", "Forecast Accuracy & Model Competition", [
                ("Accuracy by Level", forecast_result.hierarchy_accuracy),
                ("Model Leaderboard", forecast_result.model_leaderboard),
                ("SKU Accuracy", forecast_result.sku_accuracy),
                ("Category Accuracy", forecast_result.category_accuracy),
                ("All Model Scores", forecast_result.model_scores),
                ("Feature Importance", forecast_result.feature_importance),
                ("SHAP Attribution", forecast_result.shap_summary),
            ]),
            ("consensus_plan.xlsx", "Consensus Demand Plan", [
                ("Consensus Plan", consensus_result.consensus_plan),
                ("Override Audit Log", consensus_result.override_log),
                ("Scenario Comparison", consensus_result.scenario_summary),
                ("Waterfall Bridge", consensus_result.waterfall),
            ]),
            ("supply_plan.xlsx", "Time-Phased Supply Plan", [
                ("Supply Plan", supply_result.supply_plan),
                ("SKU Risk Summary", supply_result.sku_summary),
                ("Inventory Policy", supply_result.inventory_policy),
                ("Stockout Register", supply_result.stockout_risks),
            ]),
            ("capacity_alerts.xlsx", "Capacity Constraint Alert Log", [
                ("Capacity Alerts", supply_result.capacity_alerts),
                ("Breaches Only", supply_result.capacity_alerts[
                    supply_result.capacity_alerts["Severity"] == "BREACH"]
                 if not supply_result.capacity_alerts.empty else pd.DataFrame()),
            ]),
            ("otif_analysis.xlsx", "OTIF Service & Six Sigma SPC", [
                ("SPC Control Chart", kpi_result.spc_chart),
                ("SPC Violations", kpi_result.spc_violations),
                ("OTIF by Supplier", kpi_result.otif_by_supplier),
                ("OTIF by Category", kpi_result.otif_by_category),
                ("OTIF by Channel", kpi_result.otif_by_channel),
                ("OTIF by Month", kpi_result.otif_by_month),
                ("Root Cause Pareto", kpi_result.root_cause_analysis),
                ("Supplier Scorecard", kpi_result.supplier_scorecard),
            ]),
            ("executive_summary.xlsx", "Executive Insight Register", [
                ("Insights", insights),
                ("Forecast Accuracy", forecast_result.hierarchy_accuracy),
                ("Scenario Comparison", consensus_result.scenario_summary),
                ("SPC Chart", kpi_result.spc_chart),
            ]),
        ]

        for filename, title, sheets in specs:
            wb = Workbook()
            wb.remove(wb.active)
            written = 0
            for sheet_name, frame in sheets:
                ws = wb.create_sheet(sheet_name[:31])
                row = s.title_block(ws, title, sheet_name, width=10)
                end = s.write_table(ws, frame, row, max_rows=20_000)
                if frame is not None and not frame.empty:
                    n = min(len(frame), 20_000)
                    xc.add_autofilter(ws, row, row + n, 1, len(frame.columns))
                    # Data bars on the first numeric column carry magnitude
                    # faster than reading digits on a 500-row table.
                    for index, column in enumerate(frame.columns, start=1):
                        if pd.api.types.is_numeric_dtype(frame[column]) and \
                                not pd.api.types.is_bool_dtype(frame[column]):
                            letter = get_column_letter(index)
                            xc.add_data_bars(ws, f"{letter}{row + 1}:{letter}{row + n}")
                            break
                ws.sheet_view.showGridLines = False
                xc.setup_print(ws, repeat_header_row=row, title=title)
                written += 1 if frame is not None and not frame.empty else 0
            path = OUTPUT_DIR / filename
            wb.save(path)
            outputs[filename] = path
            if written == 0:
                logger.warning("%s contains no populated sheets.", filename)

        logger.info("Wrote %d supporting workbooks to outputs/.", len(outputs))
        return outputs

    def export_csvs(self, forecast_result, consensus_result, supply_result,
                    kpi_result, insights: pd.DataFrame) -> List[Path]:
        """CSV mirrors of the key frames, for the dashboard and downstream outputs."""
        exports = {
            "demand_forecast.csv": forecast_result.forecast,
            "forecast_uncertainty.csv": forecast_result.uncertainty,
            "forecast_accuracy_hierarchy.csv": forecast_result.hierarchy_accuracy,
            "forecast_sku_accuracy.csv": forecast_result.sku_accuracy,
            "model_leaderboard.csv": forecast_result.model_leaderboard,
            "model_scores.csv": forecast_result.model_scores,
            "ensemble_weights.csv": forecast_result.ensemble_weights,
            "feature_importance.csv": forecast_result.feature_importance,
            "shap_summary.csv": forecast_result.shap_summary,
            "holdout_predictions.csv": forecast_result.holdout_predictions,
            "consensus_plan.csv": consensus_result.consensus_plan,
            "override_log.csv": consensus_result.override_log,
            "scenario_summary.csv": consensus_result.scenario_summary,
            "consensus_waterfall.csv": consensus_result.waterfall,
            "supply_plan.csv": supply_result.supply_plan,
            "supply_sku_summary.csv": supply_result.sku_summary,
            "capacity_alerts.csv": supply_result.capacity_alerts,
            "stockout_risks.csv": supply_result.stockout_risks,
            "inventory_policy.csv": supply_result.inventory_policy,
            "otif_by_supplier.csv": kpi_result.otif_by_supplier,
            "otif_by_category.csv": kpi_result.otif_by_category,
            "otif_by_channel.csv": kpi_result.otif_by_channel,
            "otif_by_month.csv": kpi_result.otif_by_month,
            "otif_by_sku.csv": kpi_result.otif_by_sku,
            "spc_chart.csv": kpi_result.spc_chart,
            "spc_violations.csv": kpi_result.spc_violations,
            "root_cause_analysis.csv": kpi_result.root_cause_analysis,
            "supplier_scorecard.csv": kpi_result.supplier_scorecard,
            "executive_insights.csv": insights,
        }
        from src.planning.config import PROCESSED_DIR
        written: List[Path] = []
        for name, frame in exports.items():
            if frame is None:
                continue
            path = PROCESSED_DIR / name
            frame.to_csv(path, index=False)
            written.append(path)
        logger.info("Exported %d CSV datasets to data/processed.", len(written))
        return written


def build_reports(config: AppConfig, forecast_result, consensus_result,
                  supply_result, kpi_result, insights: pd.DataFrame,
                  split_workbooks: bool = False, data=None,
                  validation_report=None,
                  full_master_data: bool = False) -> Dict[str, object]:
    """
    Build the deliverables: one chart-led workbook plus the CSV detail layer.

    The workbook is the decision layer -- eight sheets, chart-first, ranked
    exception tables capped at a screenful. The complete row-level detail lives
    in the CSV exports, which is where filtering and pivoting actually belong.
    """
    from src.planning.workbook import build_workbook

    builder = SOPPackBuilder(config)
    workbook = build_workbook(config, forecast_result, consensus_result,
                              supply_result, kpi_result, insights,
                              data=data, validation_report=validation_report,
                              full_master_data=full_master_data)
    csvs = builder.export_csvs(forecast_result, consensus_result,
                               supply_result, kpi_result, insights)
    return {"workbook": workbook, "master_workbook": workbook,
            "workbooks": {}, "csv_exports": csvs}
