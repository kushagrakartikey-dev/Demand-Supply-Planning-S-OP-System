"""
S&OP Command Center workbook — chart-led, nine sheets.

A deliberate rebuild of the reporting layer. The previous version carried thirty
sheets of tables, which is a data dump wearing a report's clothing: everything
was present and nothing was legible.

Design rules
------------
1. **One sheet per decision domain**, not one sheet per dataset. Nine sheets
   total. If a reader has to remember which tab holds a number, there are too
   many tabs.
2. **Charts first, tables second.** Each analysis sheet leads with two to four
   charts and carries at most one compact supporting table, capped at the top
   15-20 rows. Ranked exceptions are what a reader acts on; the full 24,000-row
   detail lives in the CSV exports, where filtering actually belongs.
3. **Every sheet answers one question.** Stated in the subtitle, so a reader
   knows within two seconds whether this is the sheet they want.
4. **Consistent furniture.** Same KPI strip, same chart grid, same colours in
   the same meanings on every sheet.

Sheets::

    1 Dashboard        Whole-business position on one screen
    2 Demand           Is the forecast good enough to plan on?
    3 Consensus        What did we agree to, and who changed it?
    4 Supply & Risk    What will break, when, and can we still fix it?
    5 Service & SPC    Are we delivering, and is the process stable?
    6 Action Plan      What must someone do on Monday?
    7 What-If Model    Live Excel model for interrogating one SKU
    8 Guide & Method   How to read this, and how the numbers were made
    9 Master Data      What data was consumed, and what was cleaned first
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from src.planning import excel_charts as xc
from src.planning.config import AppConfig, Keys, OUTPUT_DIR
from src.shared.utils import ensure_directories, get_logger, human_number, safe_div

logger = get_logger("workbook")

NAVY, BLUE, LIGHT = "1F3864", "2E75B6", "D9E2F3"
GREEN, AMBER, RED, GREY = "0F7B4F", "B7791F", "9C2B2B", "F2F2F2"
INK, MUTED = "1B1F2A", "6B7280"

THIN = Side(style="thin", color="D6DCE4")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# Chart grid: two charts per band, left at column A, right at column K.
CHART_LEFT, CHART_RIGHT = "A", "K"
PANEL_COL = 100          # staging block behind the live control panel
CHART_BAND = 18          # rows consumed by one band of charts
STAGE_COL = 60           # first column of the hidden chart-data staging area


class Theme:
    """One visual language, applied everywhere."""

    def __init__(self, currency: str) -> None:
        self.currency = currency
        self.money = f'"{currency}"#,##0'
        self.header_fill = PatternFill("solid", fgColor=NAVY)
        self.band_fill = PatternFill("solid", fgColor=LIGHT)
        self.header_font = Font(bold=True, color="FFFFFF", size=10)
        self.body = Font(size=10)
        self.bold = Font(bold=True, size=10)

    # -- page furniture --------------------------------------------------- #
    def title(self, ws: Worksheet, title: str, question: str) -> int:
        """Title band stating the one question this sheet answers."""
        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=18, color=NAVY)
        ws.merge_cells("A1:H1")
        ws["A2"] = question
        ws["A2"].font = Font(italic=True, size=11, color=MUTED)
        ws.merge_cells("A2:H2")
        xc.add_back_link(ws, row=1, col=19)
        ws.row_dimensions[1].height = 24
        return 4

    def band(self, ws: Worksheet, row: int, text: str, width: int = 18) -> int:
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        for col in range(1, width + 1):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=BLUE)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
        return row + 2

    def kpi_strip(self, ws: Worksheet, row: int,
                  cards: Sequence[Tuple[str, str, str, str]]) -> int:
        """
        A row of KPI tiles: (label, value, sub-caption, status).

        Four tiles per row, each three columns wide, so the strip lines up with
        the chart grid beneath it.
        """
        colours = {"good": GREEN, "warn": AMBER, "bad": RED, "neutral": NAVY}
        for index, (label, value, sub, status) in enumerate(cards):
            col = 1 + index * 3
            colour = colours.get(status, NAVY)

            head = ws.cell(row=row, column=col, value=label.upper())
            head.font = Font(bold=True, size=8, color="FFFFFF")
            head.alignment = Alignment(horizontal="center", vertical="center")
            for offset in range(3):
                ws.cell(row=row, column=col + offset).fill = PatternFill(
                    "solid", fgColor=colour)
            ws.merge_cells(start_row=row, start_column=col,
                           end_row=row, end_column=col + 2)

            body = ws.cell(row=row + 1, column=col, value=value)
            body.font = Font(bold=True, size=18, color=colour)
            body.alignment = Alignment(horizontal="center", vertical="center")
            ws.merge_cells(start_row=row + 1, start_column=col,
                           end_row=row + 1, end_column=col + 2)

            note = ws.cell(row=row + 2, column=col, value=sub)
            note.font = Font(size=8, color=MUTED)
            note.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.merge_cells(start_row=row + 2, start_column=col,
                           end_row=row + 2, end_column=col + 2)

            for offset in range(3):
                for r in (row, row + 1, row + 2):
                    ws.cell(row=r, column=col + offset).border = BORDER

        ws.row_dimensions[row].height = 16
        ws.row_dimensions[row + 1].height = 26
        ws.row_dimensions[row + 2].height = 22
        return row + 4

    def table(self, ws: Worksheet, df: Optional[pd.DataFrame], row: int,
              caption: str = "", max_rows: int = 15,
              formats: Optional[Dict[str, str]] = None,
              rag: Optional[Dict[str, Tuple[float, float]]] = None,
              reverse_rag: Optional[Dict[str, Tuple[float, float]]] = None,
              bar_col: Optional[str] = None) -> int:
        """
        A compact, ranked supporting table.

        Capped at `max_rows` on purpose. A table longer than a screen stops
        being a summary and becomes a search problem, which is what the CSV
        exports are for.
        """
        if caption:
            cell = ws.cell(row=row, column=1, value=caption)
            cell.font = Font(italic=True, size=9, color=MUTED)
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=12)
            row += 1

        if df is None or df.empty:
            cell = ws.cell(row=row, column=1,
                           value="Nothing met the criteria — no action required here.")
            cell.font = Font(italic=True, size=10, color=GREEN)
            return row + 2

        truncated = len(df) > max_rows
        view = df.head(max_rows).copy()
        for column in view.select_dtypes("float").columns:
            view[column] = view[column].round(2)

        header_row = row
        for offset, column in enumerate(view.columns):
            cell = ws.cell(row=row, column=1 + offset,
                           value=str(column).replace("_", " "))
            cell.font = self.header_font
            cell.fill = self.header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = BORDER
        row += 1

        for index, (_, record) in enumerate(view.iterrows()):
            for offset, column in enumerate(view.columns):
                value = record[column]
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = None if not np.isfinite(value) else float(value)
                elif isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                elif value is not None and not isinstance(value, (int, float, str)):
                    value = str(value)
                cell = ws.cell(row=row, column=1 + offset, value=value)
                cell.font = self.body
                cell.border = BORDER
                if index % 2:
                    cell.fill = self.band_fill
                if formats and column in formats:
                    cell.number_format = formats[column]
                elif isinstance(value, float):
                    cell.number_format = "#,##0.00"
                elif isinstance(value, int):
                    cell.number_format = "#,##0"
            row += 1

        last = row - 1
        columns = list(view.columns)
        for column, (good, warn) in (rag or {}).items():
            if column in columns:
                letter = get_column_letter(columns.index(column) + 1)
                self._rag(ws, f"{letter}{header_row + 1}:{letter}{last}", good, warn)
        for column, (bad, warn) in (reverse_rag or {}).items():
            if column in columns:
                letter = get_column_letter(columns.index(column) + 1)
                self._rag(ws, f"{letter}{header_row + 1}:{letter}{last}",
                          bad, warn, reverse=True)
        if bar_col and bar_col in columns:
            letter = get_column_letter(columns.index(bar_col) + 1)
            xc.add_data_bars(ws, f"{letter}{header_row + 1}:{letter}{last}")

        xc.add_autofilter(ws, header_row, last, 1, len(columns))
        self.autofit(ws, view)

        if truncated:
            note = ws.cell(row=row, column=1,
                           value=f"Top {max_rows} of {len(df):,} shown — ranked by the "
                                 f"column that matters. Full detail in the CSV exports.")
            note.font = Font(italic=True, size=8, color=MUTED)
            row += 1
        return row + 2

    @staticmethod
    def _rag(ws: Worksheet, cell_range: str, a: float, b: float,
             reverse: bool = False) -> None:
        from openpyxl.formatting.rule import CellIsRule
        good = PatternFill("solid", fgColor="C6EFCE")
        warn = PatternFill("solid", fgColor="FFEB9C")
        bad = PatternFill("solid", fgColor="FFC7CE")
        if reverse:                     # high is bad
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="greaterThanOrEqual", formula=[str(a)], fill=bad))
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="greaterThanOrEqual", formula=[str(b)], fill=warn))
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="lessThan", formula=[str(b)], fill=good))
        else:                           # high is good
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="greaterThanOrEqual", formula=[str(a)], fill=good))
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="greaterThanOrEqual", formula=[str(b)], fill=warn))
            ws.conditional_formatting.add(cell_range, CellIsRule(
                operator="lessThan", formula=[str(b)], fill=bad))

    @staticmethod
    def autofit(ws: Worksheet, df: pd.DataFrame, start_col: int = 1) -> None:
        for offset, column in enumerate(df.columns):
            try:
                body = int(df[column].astype(str).str.len().max())
            except Exception:
                body = 0
            width = min(max(len(str(column)) + 2, body + 2, 10), 40)
            letter = get_column_letter(start_col + offset)
            if (ws.column_dimensions[letter].width or 0) < width:
                ws.column_dimensions[letter].width = width

    @staticmethod
    def note(ws: Worksheet, row: int, text: str) -> int:
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = Font(italic=True, size=9, color=MUTED)
        cell.alignment = Alignment(wrap_text=True)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=18)
        return row + 2


class CommandCenterWorkbook:
    """Builds the eight-sheet chart-led workbook."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.report
        self.theme = Theme(self.cfg.currency_symbol)
        ensure_directories()

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    def _prepare(self, forecast, consensus, supply, kpi, insights) -> Dict[str, object]:
        scenario = consensus.diagnostics["scenario"]
        plan = consensus.consensus_plan
        primary = plan[plan["Scenario"] == scenario]

        hierarchy = forecast.hierarchy_accuracy
        review_wape = None
        if not hierarchy.empty:
            match = hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
            if not match.empty:
                review_wape = float(match.iloc[0]["WAPE"])

        fva = None
        if not hierarchy.empty:
            sku_week = hierarchy[hierarchy["Aggregation_Level"] == "SKU / Week"]
            if not sku_week.empty and np.isfinite(sku_week.iloc[0].get("FVA_vs_Naive", np.nan)):
                fva = float(sku_week.iloc[0]["FVA_vs_Naive"])

        # weekly rollups used by several charts
        forecast_weekly = (forecast.forecast.groupby("horizon", as_index=False)
                           .agg(Forecast=("Forecast_Units", "sum"),
                                Lower_80=("P10", "sum"), Upper_80=("P90", "sum")))
        forecast_weekly["Week"] = "W" + forecast_weekly["horizon"].astype(str)

        consensus_weekly = (primary.groupby("horizon", as_index=False)
                            .agg(Statistical=("Forecast_Units", "sum"),
                                 Commercial=("Commercial_Forecast", "sum"),
                                 Consensus=("Consensus_Units", "sum")))
        consensus_weekly["Week"] = "W" + consensus_weekly["horizon"].astype(str)

        supply_weekly = (supply.supply_plan.groupby("horizon", as_index=False)
                         .agg(Demand=("Demand_Units", "sum"),
                              Receipts=("Planned_Receipts", "sum"),
                              Closing_Inventory=("Closing_Inventory", "sum"),
                              Safety_Stock=("Safety_Stock", "sum")))
        supply_weekly["Week"] = "W" + supply_weekly["horizon"].astype(str)

        projected = (supply.stockout_risks[
            supply.stockout_risks["Event_Type"] == "Projected Stockout"]
            if not supply.stockout_risks.empty else pd.DataFrame())
        risk_weekly = pd.DataFrame()
        if not projected.empty:
            risk_weekly = (projected.groupby("horizon", as_index=False)
                           .agg(Revenue_At_Risk=("Revenue_At_Risk", "sum"),
                                SKUs=(Keys.SKU, "nunique")))
            risk_weekly["Week"] = "W" + risk_weekly["horizon"].astype(str)

        capacity = supply.capacity_alerts
        supplier_load = pd.DataFrame()
        if not capacity.empty:
            suppliers = capacity[capacity["Resource_Type"] == "Supplier"]
            if not suppliers.empty:
                supplier_load = (suppliers.groupby("Resource_ID", as_index=False)
                                 .agg(Peak_Utilisation=("Utilisation_Pct", "max"),
                                      Weeks_Over=("Severity",
                                                  lambda s: int((s != "OK").sum())),
                                      Overload_Units=("Overload_Units", "sum"),
                                      SKUs=("SKUs_Affected", "max"))
                                 .sort_values("Peak_Utilisation", ascending=False)
                                 .head(12))

        excess = supply.sku_summary[
            supply.sku_summary["Min_Cover"] > self.config.supply.excess_cover_weeks]

        return {
            "scenario": scenario,
            "primary": primary,
            "hierarchy": hierarchy,
            "review_wape": review_wape,
            "fva": fva,
            "forecast_weekly": forecast_weekly,
            "consensus_weekly": consensus_weekly,
            "supply_weekly": supply_weekly,
            "risk_weekly": risk_weekly,
            "supplier_load": supplier_load,
            "excess_capital": float(excess["Avg_Working_Capital"].sum())
            if not excess.empty else 0.0,
            "excess_count": int(len(excess)),
            "fdiag": forecast.diagnostics,
            "cdiag": consensus.diagnostics,
            "sdiag": supply.diagnostics,
            "kdiag": kpi.diagnostics,
        }


    # ------------------------------------------------------------------ #
    # Live control panel
    # ------------------------------------------------------------------ #
    def _control_panel(self, ws: Worksheet, row: int, ctx, supply,
                       consensus) -> int:
        """
        A filter-driven control panel computed entirely by Excel.

        Three dropdowns — category, risk band, channel — drive six KPI cells and
        a ranked top-ten table through ``SUMPRODUCT`` and ``INDEX``/``MATCH``.
        Nothing here is precomputed in Python: change a filter and Excel
        recalculates the whole panel, including the ranking.

        Why this matters more than another chart: a static report answers the
        question the analyst thought of. A filtered panel answers the question
        the reader has. That is the difference between a report and a tool, and
        it is the one thing a planner will actually use in the meeting.

        Mechanics: a per-SKU staging block sits far to the right (column 100+).
        One helper column evaluates the filter predicate to 1 or 0; a second
        multiplies it by revenue at risk and adds a row-index tiebreaker so
        ``LARGE`` can rank without ties collapsing. Every visible cell then
        aggregates or looks up against those two columns.
        """
        t = self.theme
        c = self.cfg.currency_symbol
        summary = supply.sku_summary.copy()
        if summary.empty:
            return row

        projected = supply.stockout_risks[
            supply.stockout_risks["Event_Type"] == "Projected Stockout"] \
            if not supply.stockout_risks.empty else pd.DataFrame()
        value_map = (projected.groupby(Keys.SKU)["Revenue_At_Risk"].sum()
                     if not projected.empty else pd.Series(dtype=float))
        summary["Revenue_At_Risk"] = summary[Keys.SKU].map(value_map).fillna(0.0)

        value_by_sku = (ctx["primary"].groupby(Keys.SKU)["Consensus_Value"].sum()
                        if "Consensus_Value" in ctx["primary"].columns
                        else pd.Series(dtype=float))
        summary["Consensus_Value"] = summary[Keys.SKU].map(value_by_sku).fillna(0.0)
        summary["Expedite"] = (summary["First_Stockout_Week"].notna()
                               & ~summary["Can_Recover_In_Lead_Time"].astype(bool)
                               ).astype(int)

        stage = summary[[Keys.SKU, Keys.CATEGORY, Keys.CHANNEL, "Risk_Band",
                         "Supply_Risk_Index", "Revenue_At_Risk", "Consensus_Value",
                         "Expedite"]].fillna({Keys.CHANNEL: "n/a"})
        n = len(stage)

        # -- staging block ------------------------------------------------ #
        L = {name: get_column_letter(PANEL_COL + i)
             for i, name in enumerate(["sku", "cat", "chan", "band", "risk",
                                        "rar", "value", "exp", "scope", "rank"])}
        for i, name in enumerate(stage.columns):
            ws.cell(row=1, column=PANEL_COL + i, value=str(name))
        for r, (_, rec) in enumerate(stage.iterrows(), start=2):
            for i, name in enumerate(stage.columns):
                v = rec[name]
                ws.cell(row=r, column=PANEL_COL + i,
                        value=v.item() if hasattr(v, "item") else v)

        row = t.band(ws, row, "CONTROL PANEL — FILTER THE WHOLE VIEW (LIVE EXCEL)")
        hint = ws.cell(row=row, column=5,
                       value="Change the blue cells. Every KPI and the ranked table "
                             "below recalculate inside Excel.")
        hint.font = Font(italic=True, size=9, color=MUTED)
        ws.merge_cells(start_row=row, start_column=5, end_row=row, end_column=14)

        filters = [("Category", Keys.CATEGORY, "cat"),
                   ("Risk band", "Risk_Band", "band"),
                   ("Channel", Keys.CHANNEL, "chan")]
        refs: Dict[str, str] = {}
        for i, (label, column, key) in enumerate(filters):
            r = row + i
            lab = ws.cell(row=r, column=1, value=label)
            lab.font = t.bold
            cell = ws.cell(row=r, column=3, value="(All)")
            cell.font = Font(bold=True, size=11, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center")
            cell.border = BORDER
            ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
            options = ["(All)"] + sorted(stage[column].dropna().unique().tolist())
            dv = DataValidation(type="list",
                                formula1='"' + ",".join(str(o) for o in options) + '"',
                                allow_blank=False, showDropDown=False)
            ws.add_data_validation(dv)
            dv.add(cell)
            refs[key] = f"$C${r}"
        row += len(filters) + 1

        # -- scope predicate ---------------------------------------------- #
        last = n + 1
        for r in range(2, last + 1):
            ws.cell(row=r, column=PANEL_COL + 8, value=(
                f'=IF((({refs["cat"]}="(All)")+({L["cat"]}{r}={refs["cat"]})>0)'
                f'*(({refs["band"]}="(All)")+({L["band"]}{r}={refs["band"]})>0)'
                f'*(({refs["chan"]}="(All)")+({L["chan"]}{r}={refs["chan"]})>0)>0,1,0)'))
            # Row index tiebreaker keeps LARGE() from collapsing equal values.
            ws.cell(row=r, column=PANEL_COL + 9,
                    value=f'={L["scope"]}{r}*{L["rar"]}{r}+ROW()/1000000')

        scope = f'{L["scope"]}$2:{L["scope"]}${last}'
        rank = f'{L["rank"]}$2:{L["rank"]}${last}'

        def kpi(label: str, formula: str, fmt: str, sub: str, col: int) -> None:
            head = ws.cell(row=row, column=col, value=label.upper())
            head.font = Font(bold=True, size=8, color="FFFFFF")
            head.alignment = Alignment(horizontal="center")
            for o in range(3):
                ws.cell(row=row, column=col + o).fill = PatternFill(
                    "solid", fgColor=NAVY)
                ws.cell(row=row, column=col + o).border = BORDER
            ws.merge_cells(start_row=row, start_column=col,
                           end_row=row, end_column=col + 2)
            body = ws.cell(row=row + 1, column=col, value=formula)
            body.font = Font(bold=True, size=16, color=NAVY)
            body.number_format = fmt
            body.alignment = Alignment(horizontal="center")
            body.border = BORDER
            ws.merge_cells(start_row=row + 1, start_column=col,
                           end_row=row + 1, end_column=col + 2)
            note = ws.cell(row=row + 2, column=col, value=sub)
            note.font = Font(size=8, color=MUTED)
            note.alignment = Alignment(horizontal="center")
            ws.merge_cells(start_row=row + 2, start_column=col,
                           end_row=row + 2, end_column=col + 2)

        kpi("SKUs in scope", f"=SUMPRODUCT({scope})", "#,##0",
            "responds to all three filters", 1)
        kpi("Consensus value", f'=SUMPRODUCT({scope},{L["value"]}$2:{L["value"]}${last})',
            t.money, "agreed plan in scope", 4)
        kpi("Revenue at risk", f'=SUMPRODUCT({scope},{L["rar"]}$2:{L["rar"]}${last})',
            t.money, "projected stockouts", 7)
        row += 4
        kpi("Red-risk SKUs",
            f'=SUMPRODUCT({scope},--({L["band"]}$2:{L["band"]}${last}="Red"))',
            "#,##0", "risk index >= 65", 1)
        kpi("Expedite required",
            f'=SUMPRODUCT({scope},{L["exp"]}$2:{L["exp"]}${last})', "#,##0",
            "inside supplier lead time", 4)
        kpi("Avg risk index",
            f'=IFERROR(SUMPRODUCT({scope},{L["risk"]}$2:{L["risk"]}${last})'
            f'/SUMPRODUCT({scope}),0)', "0.0", "0-100 scale", 7)
        row += 4

        # -- live ranked table -------------------------------------------- #
        cap = ws.cell(row=row, column=1,
                      value="TOP 10 SKUs BY REVENUE AT RISK — ranked live from the "
                            "filters above")
        cap.font = Font(bold=True, size=10, color=NAVY)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=12)
        row += 1
        for i, head in enumerate(["#", "SKU", "Category", "Risk band",
                                  "Risk index", "Revenue at risk"]):
            cell = ws.cell(row=row, column=1 + i, value=head)
            cell.font = t.header_font
            cell.fill = t.header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = BORDER
        row += 1

        for k in range(1, 11):
            guard = f'IF(SUMPRODUCT({scope})<{k},"",'
            pos = f'MATCH(LARGE({rank},{k}),{rank},0)'
            ws.cell(row=row, column=1, value=k).font = t.body
            for i, source in enumerate(["sku", "cat", "band", "risk", "rar"]):
                col_range = f'{L[source]}$2:{L[source]}${last}'
                cell = ws.cell(row=row, column=2 + i,
                               value=f'={guard}IFERROR(INDEX({col_range},{pos}),""))')
                cell.font = t.body
                cell.border = BORDER
                if source == "rar":
                    cell.number_format = t.money
                elif source == "risk":
                    cell.number_format = "0.0"
            row += 1

        row = t.note(ws, row + 1,
                     "Every figure in this panel is an Excel formula over a staged "
                     "per-SKU block (far right, column CV onward). Nothing is "
                     "precomputed — the ranking re-sorts as you filter.")
        return row
    # ------------------------------------------------------------------ #
    # Sheet 1 — Dashboard
    # ------------------------------------------------------------------ #
    def _dashboard(self, ws: Worksheet, ctx, forecast, consensus, supply,
                   kpi, insights) -> None:
        t = self.theme
        c = self.cfg.currency_symbol
        sd, cd, kd = ctx["sdiag"], ctx["cdiag"], ctx["kdiag"]

        ws["A1"] = "S&OP COMMAND CENTER"
        ws["A1"].font = Font(bold=True, size=24, color=NAVY)
        ws.merge_cells("A1:H1")
        ws["A2"] = (f"Demand & Supply Planning  ·  {ctx['scenario']} scenario  ·  "
                    f"12-week horizon  ·  "
                    f"{datetime.now(timezone.utc).strftime('%d %b %Y')}")
        ws["A2"].font = Font(size=11, color=MUTED)
        ws.merge_cells("A2:H2")
        ws["A3"] = ("Synthetic / simulated data for portfolio demonstration. "
                    "Not employer or client data.")
        ws["A3"].font = Font(italic=True, size=9, color=MUTED)
        ws.merge_cells("A3:H3")

        row = 5
        otif = kd["overall_otif_pct"]
        wape = ctx["review_wape"]
        red = int(sd.get("risk_distribution", {}).get("Red", 0))

        row = t.kpi_strip(ws, row, [
            ("Consensus Value", human_number(cd["consensus_value"], c),
             f"{ctx['scenario']} plan", "neutral"),
            ("Forecast Accuracy", f"{wape:.1f}%" if wape else "n/a",
             "WAPE at review level", "good" if wape and wape < 15 else "warn"),
            ("Service — OTIF", f"{otif:.1f}%", kd["benchmark"],
             "good" if otif >= 95 else "warn" if otif >= 90 else "bad"),
            ("Forecast Value Added",
             f"+{ctx['fva']:.0f} pts" if ctx["fva"] else "n/a",
             "vs seasonal naive", "good"),
        ])
        row = t.kpi_strip(ws, row, [
            ("Revenue At Risk", human_number(sd.get("revenue_at_risk", 0), c),
             "projected stockouts", "bad"),
            ("Expedite Now", f"{sd.get('expedite_required', 0):,}",
             "inside lead time", "bad" if sd.get("expedite_required") else "good"),
            ("Capacity Breaches", f"{sd.get('capacity_breaches', 0):,}",
             "resource-weeks >100%",
             "bad" if sd.get("capacity_breaches") else "good"),
            ("Red-Risk SKUs", f"{red:,}",
             f"of {sd.get('skus_planned', 0):,}", "warn" if red else "good"),
        ])

        row = self._control_panel(ws, row, ctx, supply, consensus)

        row = t.band(ws, row, "PERFORMANCE — WHERE THE BUSINESS STANDS")
        xc.add_accuracy_ladder_chart(
            ws, ctx["hierarchy"], f"{CHART_LEFT}{row}", data_col=STAGE_COL,
            benchmark=self.config.forecast.portfolio_mape_target)
        xc.add_spc_control_chart(ws, kpi.spc_chart, f"{CHART_RIGHT}{row}",
                                 data_col=STAGE_COL + 8, width=16, height=8.5)
        row += CHART_BAND

        row = t.band(ws, row, "RISK — WHAT WILL BREAK, AND WHEN")
        xc.add_risk_distribution_chart(
            ws, sd.get("risk_distribution", {}), f"{CHART_LEFT}{row}",
            data_col=STAGE_COL + 16)
        if not ctx["risk_weekly"].empty:
            xc.add_grouped_bar_chart(
                ws, ctx["risk_weekly"], "Week", ["Revenue_At_Risk"],
                "Revenue at Risk by Week — Notice Available",
                f"{CHART_RIGHT}{row}", y_title=f"Value ({c})",
                data_col=STAGE_COL + 20)
        row += CHART_BAND

        row = t.band(ws, row, "PLAN — DEMAND AGREED VERSUS DEMAND DELIVERABLE")
        xc.add_waterfall_chart(ws, consensus.waterfall, f"{CHART_LEFT}{row}",
                               data_col=STAGE_COL + 26)
        xc.add_line_chart(
            ws, ctx["supply_weekly"], "Week",
            ["Demand", "Receipts", "Closing_Inventory", "Safety_Stock"],
            "Inventory Trajectory vs Safety Stock", f"{CHART_RIGHT}{row}",
            y_title="Units", data_col=STAGE_COL + 32, dashed_last=True)
        row += CHART_BAND + 2

        row = t.band(ws, row, "TOP DECISIONS THIS CYCLE")
        if insights is not None and not insights.empty:
            columns = [c2 for c2 in ["Severity", "Domain", "Headline", "Owner"]
                       if c2 in insights.columns]
            top = insights[insights["Severity"].isin(["Critical", "High"])][columns]
            row = t.table(ws, top if not top.empty else insights[columns], row,
                          max_rows=8)
        t.note(ws, row,
               "Full detail on the tabs: Demand · Consensus · Supply & Risk · "
               "Service & SPC · Action Plan. Sheet 7 is a live Excel model.")
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Dashboard")

    # ------------------------------------------------------------------ #
    # Sheet 2 — Demand
    # ------------------------------------------------------------------ #
    def _demand(self, ws: Worksheet, ctx, forecast) -> None:
        t = self.theme
        fd = ctx["fdiag"]
        row = t.title(ws, "DEMAND & FORECAST",
                      "Is the forecast good enough to plan on?")

        coverage = fd.get("interval_coverage", {})
        mix = (forecast.sku_accuracy["Selected_Model"].value_counts()
               if not forecast.sku_accuracy.empty else pd.Series(dtype=int))
        row = t.kpi_strip(ws, row, [
            ("Review-Level WAPE",
             f"{ctx['review_wape']:.1f}%" if ctx["review_wape"] else "n/a",
             "Category / Month", "good" if ctx["review_wape"] and
             ctx["review_wape"] < 15 else "warn"),
            ("SKU-Week WAPE",
             f"{fd.get('portfolio_holdout_wape', 0):.1f}%",
             "hardest cut", "warn"),
            ("Champion Model", str(mix.index[0]) if len(mix) else "n/a",
             f"{mix.iloc[0]:,} of {mix.sum():,} SKUs" if len(mix) else "", "neutral"),
            ("PI 80% Coverage",
             f"{coverage.get('coverage_80', float('nan')):.0f}%",
             "target 80%", "good"),
        ])

        row = t.band(ws, row, "MODEL PERFORMANCE")
        xc.add_accuracy_ladder_chart(
            ws, ctx["hierarchy"], f"{CHART_LEFT}{row}", data_col=STAGE_COL,
            benchmark=self.config.forecast.portfolio_mape_target)
        leaderboard = forecast.model_leaderboard
        if not leaderboard.empty:
            holdout = leaderboard[(leaderboard["Stage"] == "HOLDOUT")
                                  & (leaderboard["Tier"] == "Competition Panel")]
            if holdout.empty:
                holdout = leaderboard[leaderboard["Stage"] == "HOLDOUT"]
            xc.add_grouped_bar_chart(
                ws, holdout.sort_values("Median_MAPE"), "Model",
                ["Median_MAPE"], "Model Competition — Untouched Holdout",
                f"{CHART_RIGHT}{row}", y_title="Median MAPE (%)",
                data_col=STAGE_COL + 8)
        row += CHART_BAND

        row = t.band(ws, row, "FORECAST PROFILE AND CATEGORY BIAS")
        xc.add_area_band_chart(
            ws, ctx["forecast_weekly"], "Week", "Lower_80", "Upper_80", "Forecast",
            "12-Week Forecast with 80% Prediction Interval",
            f"{CHART_LEFT}{row}", y_title="Units", data_col=STAGE_COL + 14)
        category = forecast.category_accuracy
        if not category.empty and "Weighted_Bias_Pct" in category.columns:
            xc.add_grouped_bar_chart(
                ws, category.sort_values("Weighted_Bias_Pct"), Keys.CATEGORY,
                ["Weighted_Bias_Pct"],
                "Forecast Bias by Category (+ over-forecast, − under)",
                f"{CHART_RIGHT}{row}", y_title="Bias (%)", data_col=STAGE_COL + 20)
        row += CHART_BAND

        row = t.band(ws, row, "ACCURACY BY AGGREGATION LEVEL")
        row = t.table(
            ws, ctx["hierarchy"], row,
            "Error cancels as demand aggregates. Quoting a single accuracy number "
            "without its level is meaningless — this is the honest full picture.",
            max_rows=10,
            reverse_rag={"WAPE": (30, 15), "MAPE": (35, 20)})

        row = t.band(ws, row, "HARDEST SKUs TO FORECAST")
        accuracy = forecast.sku_accuracy
        if not accuracy.empty:
            columns = [c for c in [Keys.SKU, Keys.CATEGORY, "Selected_Model",
                                   "MAPE", "WAPE", "Bias_Pct", "Total_Actual"]
                       if c in accuracy.columns]
            worst = accuracy.sort_values("MAPE", ascending=False)[columns]
            row = t.table(ws, worst, row,
                          "Plan these on commercial judgement with wider buffers, "
                          "not on the statistical baseline.",
                          max_rows=15, reverse_rag={"MAPE": (50, 25)},
                          bar_col="Total_Actual")
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Demand")

    # ------------------------------------------------------------------ #
    # Sheet 3 — Consensus
    # ------------------------------------------------------------------ #
    def _consensus(self, ws: Worksheet, ctx, consensus) -> None:
        t = self.theme
        c = self.cfg.currency_symbol
        cd = ctx["cdiag"]
        row = t.title(ws, "CONSENSUS PLAN",
                      "What did we agree to, and who changed the number?")

        statistical = ctx["primary"]["Forecast_Units"].sum()
        uplift = safe_div(cd["consensus_units"] - statistical, statistical) * 100
        row = t.kpi_strip(ws, row, [
            ("Consensus Volume", f"{cd['consensus_units'] / 1e6:.2f}M",
             "units over horizon", "neutral"),
            ("Uplift vs Statistical", f"{uplift:+.1f}%",
             "commercial + constraint", "warn" if abs(uplift) > 12 else "good"),
            ("Override Alerts", f"{cd['override_alerts']:,}",
             ">20% deviation", "warn" if cd["override_alerts"] else "good"),
            ("Constrained Out", human_number(cd["constrained_out_value"], c),
             "capacity cannot serve", "bad" if cd["constrained_out_value"] else "good"),
        ])

        row = t.band(ws, row, "HOW THE NUMBER WAS BUILT")
        xc.add_waterfall_chart(ws, consensus.waterfall, f"{CHART_LEFT}{row}",
                               data_col=STAGE_COL)
        summary = consensus.scenario_summary
        if not summary.empty:
            xc.add_grouped_bar_chart(
                ws, summary, "Scenario",
                ["Consensus_Units", "Constrained_Out_Units"],
                "Scenario Comparison — Achievable vs Lost", f"{CHART_RIGHT}{row}",
                y_title="Units", data_col=STAGE_COL + 8)
        row += CHART_BAND

        row = t.band(ws, row, "WEEKLY VIEW — BASELINE, COMMERCIAL, AGREED")
        xc.add_line_chart(
            ws, ctx["consensus_weekly"], "Week",
            ["Statistical", "Commercial", "Consensus"],
            "Statistical Baseline vs Commercial View vs Final Consensus",
            f"{CHART_LEFT}{row}", y_title="Units", data_col=STAGE_COL + 14)
        row += CHART_BAND

        row = t.band(ws, row, "SCENARIO COMPARISON")
        row = t.table(ws, summary, row,
                      "Scenario choice is a working-capital and service decision, "
                      "not a forecasting one.", max_rows=5,
                      formats={"Consensus_Value": t.money,
                               "Constrained_Out_Value": t.money})

        row = t.band(ws, row, "LARGEST COMMERCIAL OVERRIDES — GOVERNANCE")
        log = consensus.override_log
        if not log.empty:
            columns = [c2 for c2 in [Keys.SKU, Keys.CATEGORY, "Statistical_Forecast",
                                     "Commercial_Override", "Override_Deviation_Pct",
                                     "Commercial_Driver", "Submitted_By",
                                     "Governance_Status"] if c2 in log.columns]
            row = t.table(
                ws, log[columns], row,
                "The statistical baseline is never overwritten — it is carried "
                "alongside so a plan miss can be attributed to the model or to the "
                "override. Anything beyond ±20% requires written justification.",
                max_rows=15)
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Consensus")

    # ------------------------------------------------------------------ #
    # Sheet 4 — Supply & Risk
    # ------------------------------------------------------------------ #
    def _supply(self, ws: Worksheet, ctx, supply) -> None:
        t = self.theme
        c = self.cfg.currency_symbol
        sd = ctx["sdiag"]
        row = t.title(ws, "SUPPLY & RISK",
                      "What will break, when, and can we still fix it?")

        row = t.kpi_strip(ws, row, [
            ("Projected Stockouts", f"{sd.get('skus_with_stockout', 0):,}",
             f"earliest week {sd.get('earliest_stockout_week', '—')}", "bad"),
            ("Expedite Required", f"{sd.get('expedite_required', 0):,}",
             "inside lead time", "bad"),
            ("Capacity Breaches", f"{sd.get('capacity_breaches', 0):,}",
             "resource-weeks", "bad" if sd.get("capacity_breaches") else "good"),
            ("Working Capital",
             human_number(sd.get("total_working_capital", 0), c),
             "week 1 inventory", "neutral"),
        ])

        row = t.band(ws, row, "INVENTORY AND CAPACITY")
        xc.add_line_chart(
            ws, ctx["supply_weekly"], "Week",
            ["Demand", "Receipts", "Closing_Inventory", "Safety_Stock"],
            "Inventory Trajectory vs Safety Stock", f"{CHART_LEFT}{row}",
            y_title="Units", data_col=STAGE_COL, dashed_last=True)
        if not ctx["supplier_load"].empty:
            xc.add_grouped_bar_chart(
                ws, ctx["supplier_load"], "Resource_ID", ["Peak_Utilisation"],
                "Supplier Peak Capacity Utilisation (%) — worst 12",
                f"{CHART_RIGHT}{row}", y_title="Utilisation %",
                data_col=STAGE_COL + 8, horizontal=True)
        row += CHART_BAND

        row = t.band(ws, row, "RISK PROFILE")
        xc.add_risk_distribution_chart(
            ws, sd.get("risk_distribution", {}), f"{CHART_LEFT}{row}",
            data_col=STAGE_COL + 14)
        if not ctx["risk_weekly"].empty:
            xc.add_grouped_bar_chart(
                ws, ctx["risk_weekly"], "Week", ["SKUs"],
                "SKUs Stocking Out by Week — Advance Notice",
                f"{CHART_RIGHT}{row}", y_title="SKUs", data_col=STAGE_COL + 18)
        row += CHART_BAND

        row = t.band(ws, row, "EXPEDITE LIST — CANNOT BE FIXED BY A NORMAL ORDER")
        summary = supply.sku_summary
        urgent = summary[summary["First_Stockout_Week"].notna()
                         & ~summary["Can_Recover_In_Lead_Time"].astype(bool)].copy()
        if not urgent.empty:
            projected = supply.stockout_risks[
                supply.stockout_risks["Event_Type"] == "Projected Stockout"]
            value_map = (projected.groupby(Keys.SKU)["Revenue_At_Risk"].sum()
                         if not projected.empty else pd.Series(dtype=float))
            urgent["Revenue_At_Risk"] = urgent[Keys.SKU].map(value_map).fillna(0.0)
            urgent = urgent.sort_values("Revenue_At_Risk", ascending=False)
        columns = [c2 for c2 in [Keys.SKU, Keys.CATEGORY, Keys.SUPPLIER, "ABC_Class",
                                 "First_Stockout_Week", "Lead_Time_Weeks",
                                 "Revenue_At_Risk", "Supply_Risk_Index"]
                   if c2 in urgent.columns]
        row = t.table(
            ws, urgent[columns] if not urgent.empty else None, row,
            "These stock out INSIDE the supplier lead time. An order placed today "
            "arrives after the shortage starts — air-freight, alternate supply or "
            "inter-warehouse reallocation are the only levers left.",
            max_rows=15, formats={"Revenue_At_Risk": t.money},
            reverse_rag={"Supply_Risk_Index": (65, 35)}, bar_col="Revenue_At_Risk")

        row = t.band(ws, row, "SUPPLIER CAPACITY ESCALATIONS")
        row = t.table(ws, ctx["supplier_load"], row,
                      "Capacity breaches exist only in aggregate — no single SKU "
                      "looks wrong on its own.", max_rows=12,
                      reverse_rag={"Peak_Utilisation": (100, 95)})
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Supply & Risk")

    # ------------------------------------------------------------------ #
    # Sheet 5 — Service & SPC
    # ------------------------------------------------------------------ #
    def _service(self, ws: Worksheet, ctx, kpi) -> None:
        t = self.theme
        kd = ctx["kdiag"]
        row = t.title(ws, "SERVICE & STATISTICAL PROCESS CONTROL",
                      "Are we delivering, and is the process actually stable?")

        otif = kd["overall_otif_pct"]
        row = t.kpi_strip(ws, row, [
            ("OTIF", f"{otif:.1f}%", kd["benchmark"],
             "good" if otif >= 95 else "warn" if otif >= 90 else "bad"),
            ("On Time", f"{kd['on_time_pct']:.1f}%", "by promised date", "neutral"),
            ("In Full", f"{kd['in_full_pct']:.1f}%", "≥98% of qty", "neutral"),
            ("Special-Cause Months", f"{kd['special_cause_months']}",
             "genuine process shifts",
             "bad" if kd["special_cause_months"] else "good"),
        ])

        row = t.band(ws, row, "PROCESS CONTROL — SIGNAL VERSUS NOISE")
        xc.add_spc_control_chart(ws, kpi.spc_chart, f"{CHART_LEFT}{row}",
                                 data_col=STAGE_COL, width=17, height=9)
        row = t.note(
            ws, row + CHART_BAND,
            "Sigma is estimated from the average moving range (MRbar / 1.128), not "
            "the sample standard deviation, which would be inflated by the very "
            "special causes the chart exists to detect. Points inside the limits "
            "are routine variation — reacting to them adds variation rather than "
            "removing it.")

        row = t.band(ws, row, "ROOT CAUSE AND DIMENSIONAL PERFORMANCE")
        xc.add_pareto_chart(ws, kpi.root_cause_analysis, f"{CHART_LEFT}{row}",
                            data_col=STAGE_COL + 8)
        category = kpi.otif_by_category
        if not category.empty:
            xc.add_grouped_bar_chart(
                ws, category.sort_values("OTIF_Pct"), Keys.CATEGORY, ["OTIF_Pct"],
                "OTIF by Category (%)", f"{CHART_RIGHT}{row}",
                y_title="OTIF %", data_col=STAGE_COL + 14, horizontal=True)
        row += CHART_BAND

        row = t.band(ws, row, "OTIF BY CHANNEL")
        channel = kpi.otif_by_channel
        if not channel.empty:
            xc.add_grouped_bar_chart(
                ws, channel.sort_values("OTIF_Pct"), Keys.CHANNEL, ["OTIF_Pct"],
                "OTIF by Channel (%)", f"{CHART_LEFT}{row}",
                y_title="OTIF %", data_col=STAGE_COL + 20, horizontal=True)
        row += CHART_BAND

        row = t.band(ws, row, "ROOT CAUSES RANKED BY VALUE AT RISK")
        row = t.table(ws, kpi.root_cause_analysis, row,
                      "Ranked by value, not frequency — the most common failure is "
                      "rarely the most expensive.", max_rows=8,
                      formats={"Value_At_Risk": t.money},
                      bar_col="Value_At_Risk")

        row = t.band(ws, row, "SUPPLIERS BELOW THRESHOLD")
        card = kpi.supplier_scorecard
        if not card.empty:
            reliable = (card[card["Statistically_Reliable"]]
                        if "Statistically_Reliable" in card.columns else card)
            columns = [c2 for c2 in [Keys.SUPPLIER, "Orders", "OTIF_Pct_Overall",
                                     "On_Time_Pct", "In_Full_Pct", "Trend",
                                     "Risk_Flag"] if c2 in reliable.columns]
            row = t.table(ws, reliable.sort_values("OTIF_Pct_Overall")[columns], row,
                          "Order counts shown: a supplier at 100% on three orders is "
                          "not comparable to one at 94% on four hundred.",
                          max_rows=12, rag={"OTIF_Pct_Overall": (95, 90)})
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Service & SPC")

    # ------------------------------------------------------------------ #
    # Sheet 6 — Action Plan
    # ------------------------------------------------------------------ #
    def _actions(self, ws: Worksheet, ctx, supply, kpi, insights) -> None:
        t = self.theme
        c = self.cfg.currency_symbol
        sd, cd = ctx["sdiag"], ctx["cdiag"]
        row = t.title(ws, "ACTION PLAN",
                      "What must someone actually do on Monday?")

        service_value = (float(kpi.root_cause_analysis["Value_At_Risk"].sum())
                         if not kpi.root_cause_analysis.empty else 0.0)
        exposure = pd.DataFrame([
            {"Exposure": "Revenue at risk — projected stockouts",
             "Value": round(sd.get("revenue_at_risk", 0), 0),
             "Decision": "Expedite, dual-source, or accept the lost sale",
             "Owner": "Supply Planning"},
            {"Exposure": "Demand removed by capacity constraints",
             "Value": round(cd.get("constrained_out_value", 0), 0),
             "Decision": "Secure capacity, re-phase, or formally accept",
             "Owner": "Procurement / S&OP Board"},
            {"Exposure": "Service failures — trailing 12 months",
             "Value": round(service_value, 0),
             "Decision": "Attack the vital-few root causes",
             "Owner": "Supply Chain Operations"},
            {"Exposure": "Working capital in excess-cover SKUs",
             "Value": round(ctx["excess_capital"], 0),
             "Decision": "Cut order quantities, re-phase open POs",
             "Owner": "Supply Planning / Finance"},
        ]).sort_values("Value", ascending=False)

        row = t.band(ws, row, "QUANTIFIED EXPOSURE — WHERE THE VALUE IS")
        chart_row = row
        row = t.table(ws, exposure, row, "", max_rows=6,
                      formats={"Value": t.money}, bar_col="Value")
        xc.add_grouped_bar_chart(
            ws, exposure, "Exposure", ["Value"],
            f"Exposure by Type ({c})", f"{CHART_RIGHT}{chart_row}",
            y_title=f"Value ({c})", data_col=STAGE_COL, horizontal=True)
        row = max(row, chart_row + CHART_BAND)

        row = t.band(ws, row, "PRIORITISED ACTIONS — CRITICAL AND HIGH")
        if insights is not None and not insights.empty:
            columns = [c2 for c2 in ["Rank", "Severity", "Domain", "Headline",
                                     "Recommended_Action", "Owner", "Impact_Value"]
                       if c2 in insights.columns]
            actions = insights[insights["Severity"].isin(["Critical", "High"])][columns]
            if actions.empty:
                actions = insights[columns]
            row = t.table(ws, actions, row,
                          "Generated from calculated metrics crossing configured "
                          "thresholds — none are hard-coded. A clean result produces "
                          "no action rather than a reassuring one.",
                          max_rows=20, formats={"Impact_Value": t.money})

        row = t.band(ws, row, "FULL INSIGHT REGISTER")
        if insights is not None and not insights.empty:
            columns = [c2 for c2 in ["Severity", "Domain", "Headline", "Owner"]
                       if c2 in insights.columns]
            row = t.table(ws, insights[columns], row, "", max_rows=25)

        t.note(ws, row,
               "Revenue at risk assumes lost sales rather than deferred demand, so "
               "it is an upper bound. Constrained-out demand is revenue the business "
               "is choosing not to chase — it should be an explicit board decision, "
               "not a planning by-product.")
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, title="Action Plan")

    # ------------------------------------------------------------------ #
    # Sheet 8 — Guide & Method
    # ------------------------------------------------------------------ #
    def _guide(self, ws: Worksheet, ctx, forecast) -> None:
        t = self.theme
        row = t.title(ws, "GUIDE & METHOD",
                      "How to read this workbook, and how the numbers were made.")

        row = t.band(ws, row, "WHAT EACH SHEET ANSWERS")
        guide = pd.DataFrame([
            ("1 Dashboard", "Whole-business position on one screen",
             "Start here every cycle"),
            ("2 Demand", "Is the forecast good enough to plan on?",
             "Accuracy ladder, model competition, bias"),
            ("3 Consensus", "What did we agree to, and who changed it?",
             "Waterfall, scenarios, override governance"),
            ("4 Supply & Risk", "What will break, when, can we fix it?",
             "Inventory trajectory, capacity, expedite list"),
            ("5 Service & SPC", "Are we delivering, is the process stable?",
             "Control chart, Pareto, supplier scorecard"),
            ("6 Action Plan", "What must someone do on Monday?",
             "Exposure ranked by value, prioritised actions"),
            ("7 What-If Model", "What if I change this assumption?",
             "LIVE Excel formulas — edit the yellow cells"),
            ("8 Guide & Method", "How do I read and trust this?", "You are here"),
            ("9 Master Data", "What data did this run consume?",
             "Raw inventory, extract, cleaning log, cleaned master"),
        ], columns=["Sheet", "Question it answers", "Contents"])
        row = t.table(ws, guide, row, max_rows=10)

        row = t.band(ws, row, "KEY FORMULAS")
        formulas = pd.DataFrame([
            ("WAPE", "100 × Σ|A−F| / Σ|A|",
             "Volume-weighted forecast error. The metric to steer on — MAPE "
             "explodes on low-volume SKUs."),
            ("Forecast Value Added", "MAPE(seasonal naive) − MAPE(model)",
             "Percentage points the modelling adds over repeating last year. "
             "Negative would mean it destroys value."),
            ("Safety Stock", "Z × √(LT×σD² + μD²×σLT²)",
             "Includes lead-time variance, and σD is FORECAST ERROR std — the "
             "buffer absorbs error in the plan, not total demand variation."),
            ("Reorder Point", "μD × LT + safety stock", "Continuous review policy."),
            ("Weeks of Cover", "closing inventory / forward 4-week demand",
             "Capped at 52 so near-zero demand cannot return infinity."),
            ("Supply Risk Index",
             "0.32×proximity + 0.22×capacity + 0.18×cover + 0.14×lead time "
             "+ 0.14×uncertainty",
             "0–100. Green <35, Amber 35–65, Red ≥65."),
            ("OTIF", "on-time AND in-full orders / eligible orders",
             "Recomputed from promised date and ordered qty, never taken from "
             "the source system's own flag."),
            ("SPC sigma", "MRbar / 1.128",
             "Moving-range estimator, not sample std, which would be inflated "
             "by the special causes the chart detects."),
        ], columns=["Metric", "Formula", "Why it is defined this way"])
        row = t.table(ws, formulas, row, max_rows=10)

        row = t.band(ws, row, "METHOD — THE DECISIONS THAT MATTER")
        method = pd.DataFrame([
            ("Data provenance",
             "All data is SYNTHETIC, generated programmatically for portfolio "
             "demonstration. Not from any employer, client or real organisation."),
            ("Validation design",
             "Rolling-origin CV folds drive model selection and ensemble weighting. "
             "The final 12 weeks are opened once, after every decision is frozen, so "
             "reported accuracy is unbiased."),
            ("Why accuracy differs by level",
             "Forecast error cancels as demand aggregates. A +30% miss on one "
             "SKU-week and a −28% miss on the next net out at category level."),
            ("Models",
             "Holt-Winters (log space), SARIMA, Prophet with Indian festival "
             "regressors, global XGBoost (direct multi-horizon, 47 features), plus "
             "an inverse-MAPE ensemble and a seasonal-naive benchmark."),
            ("Model tiers",
             "SARIMA and Prophet run on a revenue-and-category stratified panel, "
             "not all 500 SKUs. Panel results are never extrapolated."),
            ("Capacity is shared",
             "Supplier capacity is allocated pro-rata across that supplier's SKUs. "
             "Applying full capacity to each SKU independently would overstate "
             "deliverable supply by the number of SKUs per supplier."),
            ("Where the detail lives",
             "This workbook is the decision layer. All 24,000+ rows of underlying "
             "detail are in data/processed/ as 31 CSV files."),
        ], columns=["Topic", "Note"])
        row = t.table(ws, method, max_rows=10, row=row)

        row = t.band(ws, row, "THIS RUN")
        fd, sd, kd = ctx["fdiag"], ctx["sdiag"], ctx["kdiag"]
        coverage = fd.get("interval_coverage", {})
        manifest = pd.DataFrame([
            ("Generated (UTC)",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
            ("Random seed", str(self.config.seed)),
            ("Scenario", ctx["scenario"]),
            ("SKUs forecast", f"{fd.get('n_skus', 0):,}"),
            ("History", f"{fd.get('n_weeks', 0)} weeks"),
            ("CV folds", str(fd.get("cv_folds", 0))),
            ("Holdout WAPE (SKU-week)",
             f"{fd.get('portfolio_holdout_wape', 0):.2f}%"),
            ("Review-level WAPE",
             f"{ctx['review_wape']:.2f}%" if ctx["review_wape"] else "n/a"),
            ("PI coverage 80% / 95%",
             f"{coverage.get('coverage_80', 0):.1f}% / "
             f"{coverage.get('coverage_95', 0):.1f}%"),
            ("OTIF", f"{kd.get('overall_otif_pct', 0):.2f}%"),
            ("SPC control limits",
             f"[{kd.get('lcl', 0):.2f}%, {kd.get('ucl', 0):.2f}%]"),
            ("Capacity breaches", f"{sd.get('capacity_breaches', 0):,}"),
            ("Expedite required", f"{sd.get('expedite_required', 0):,}"),
        ], columns=["Item", "Value"])
        t.table(ws, manifest, row, max_rows=15)
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, landscape=False, title="Guide & Method")

    def _master_data(self, ws: Worksheet, data, validation_report,
                     full: bool) -> None:
        """
        Sheet 9 — the lineage from raw extract to the cleaned master.

        A planning number is only as trustworthy as the data underneath it, so
        this sheet carries the audit trail: what landed, what was checked, what
        was corrected, and what the downstream modules actually consumed.

        Extracts are sampled to 25 rows by default. ``--full-master-data``
        writes every row, which is what a data-quality review wants and what a
        sub-500 KB workbook does not.
        """
        t = self.theme
        row = t.title(ws, "MASTER DATA",
                      "What data did this run consume, and what was cleaned "
                      "before it was used?")

        rows_cap = 1_000_000 if full else 25
        frames: Dict[str, pd.DataFrame] = data.as_dict() if data is not None else {}

        note = ws.cell(
            row=row, column=1,
            value=("All data on this sheet is SYNTHETIC, generated from seed "
                   f"{self.config.seed}. It is not employer or client data."))
        note.font = Font(italic=True, size=9, color=MUTED)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=12)
        row += 2

        # -- 1. Raw inventory ------------------------------------------------ #
        row = t.band(ws, row, "1. RAW DATA INVENTORY  (what the run ingested)")
        inventory = pd.DataFrame([
            {
                "Dataset": name,
                "Rows": int(len(frame)),
                "Columns": int(frame.shape[1]),
                "Memory_MB": round(
                    float(frame.memory_usage(deep=True).sum()) / 1e6, 2),
                "Leading_Fields": ", ".join(str(c) for c in frame.columns[:3]),
            }
            for name, frame in frames.items()
        ])

        if not inventory.empty:
            xc.add_grouped_bar_chart(
                ws, inventory.sort_values("Rows", ascending=False),
                category_col="Dataset", value_cols=["Rows"],
                title="Rows per source dataset", anchor=f"{CHART_LEFT}{row}",
                y_title="Rows", data_col=STAGE_COL, horizontal=True)
            xc.add_grouped_bar_chart(
                ws, inventory.sort_values("Memory_MB", ascending=False),
                category_col="Dataset", value_cols=["Memory_MB"],
                title="In-memory footprint by dataset",
                anchor=f"{CHART_RIGHT}{row}", y_title="MB",
                data_col=STAGE_COL + 8, horizontal=True)
            row += CHART_BAND

        row = t.table(ws, inventory, row,
                      caption="Every table the pipeline read, at the grain it "
                              "was generated.",
                      max_rows=30, formats={"Rows": "#,##0"})

        # -- 2. Raw extract -------------------------------------------------- #
        row = t.band(ws, row, "2. RAW EXTRACT  (as landed, before cleaning)")
        raw = frames.get("sales_history")
        row = t.table(
            ws, raw, row,
            caption=("sales_history as generated — the transactional spine. "
                     f"{'All rows.' if full else 'Sampled to 25 rows; pass --full-master-data for every row.'}"),
            max_rows=rows_cap)

        # -- 3. Cleaning & validation log ------------------------------------ #
        row = t.band(ws, row, "3. CLEANING & VALIDATION LOG")
        rules = pd.DataFrame([
            ("Week alignment", "All weekly grains snapped to Monday-start weeks",
             "Prevents a partial trailing week distorting seasonality"),
            ("Censoring guard", "Sales_Units asserted <= Demand_Units on every row",
             "Sales is the censored series an ERP records; demand is latent"),
            ("Referential integrity", "SKU / supplier / warehouse keys checked "
                                      "against their masters",
             "An orphan key silently drops volume at the first join"),
            ("Non-negativity", "Units, prices and inventory asserted >= 0",
             "A negative unit is a generator bug, not a business event"),
            ("Duplicate grain", "SKU x week asserted unique in sales history",
             "A duplicated grain double-counts demand through every stage"),
        ], columns=["Check", "Rule Applied", "Why It Matters"])
        row = t.table(ws, rules, row, max_rows=10)

        issues = validation_report
        if issues is None or (hasattr(issues, "empty") and issues.empty):
            clean = ws.cell(
                row=row, column=1,
                value="Validation completed with zero errors and zero warnings.")
            clean.font = Font(italic=True, size=10, color=GREEN)
            row += 2
        else:
            row = t.table(ws, issues, row,
                          caption="Issues raised by the validator on this run.",
                          max_rows=25)

        # -- 4. Cleaned master ------------------------------------------------ #
        row = t.band(ws, row, "4. CLEANED MASTER  (what downstream modules consumed)")
        master = frames.get("sku_master")
        t.table(
            ws, master, row,
            caption=("sku_master after validation — the dimension every other "
                     "sheet joins to."),
            max_rows=rows_cap)

        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, landscape=True, title="Master Data")

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, forecast, consensus, supply, kpi, insights,
              data=None, validation_report=None,
              full_master_data: bool = False) -> Path:
        ctx = self._prepare(forecast, consensus, supply, kpi, insights)

        wb = Workbook()
        wb.remove(wb.active)

        specs = [
            ("1 Dashboard", NAVY,
             lambda ws: self._dashboard(ws, ctx, forecast, consensus, supply,
                                        kpi, insights)),
            ("2 Demand", BLUE, lambda ws: self._demand(ws, ctx, forecast)),
            ("3 Consensus", GREEN, lambda ws: self._consensus(ws, ctx, consensus)),
            ("4 Supply & Risk", AMBER, lambda ws: self._supply(ws, ctx, supply)),
            ("5 Service & SPC", RED, lambda ws: self._service(ws, ctx, kpi)),
            ("6 Action Plan", "7030A0",
             lambda ws: self._actions(ws, ctx, supply, kpi, insights)),
        ]
        for name, colour, builder in specs:
            ws = wb.create_sheet(name)
            ws.sheet_properties.tabColor = colour
            builder(ws)

        # Sheet 7 needs a policy table to reference, so a compact one is written
        # to a hidden support area on the what-if sheet itself.
        policy_ws = wb.create_sheet("7 What-If Model")
        policy_ws.sheet_properties.tabColor = "00838F"

        guide_ws = wb.create_sheet("8 Guide & Method")
        guide_ws.sheet_properties.tabColor = "A6A6A6"
        self._guide(guide_ws, ctx, forecast)

        master_ws = wb.create_sheet("9 Master Data")
        master_ws.sheet_properties.tabColor = "5B6770"
        self._master_data(master_ws, data, validation_report, full_master_data)

        # Policy data staged on a support sheet, then the model built against it.
        support = wb.create_sheet("_policy_data")
        support.sheet_state = "hidden"
        policy = supply.inventory_policy
        keep = [c for c in [Keys.SKU, Keys.CATEGORY, "ABC_Class",
                            "Avg_Weekly_Forecast", "Demand_Std_Weekly",
                            "Lead_Time_Weeks", "Lead_Time_Std_Weeks",
                            "Service_Level_Target", "Unit_Cost", "Unit_Price",
                            "MOQ", "Safety_Stock_Units", "Reorder_Point_Units"]
                if c in policy.columns]
        compact = policy[keep].copy()
        for offset, column in enumerate(compact.columns):
            support.cell(row=1, column=1 + offset, value=column)
        for r, (_, record) in enumerate(compact.iterrows(), start=2):
            for offset, column in enumerate(compact.columns):
                value = record[column]
                if hasattr(value, "item"):
                    value = value.item()
                support.cell(row=r, column=1 + offset, value=value)

        from src.planning.what_if_model import WhatIfModelBuilder
        layout = {"header_row": 1, "first_row": 2,
                  "last_row": 1 + len(compact), "columns": list(compact.columns)}
        WhatIfModelBuilder(self, sheet_name="_policy_data").build(
            policy_ws, wb, layout)

        # Navigation strip on every sheet, pointing at the dashboard.
        names = [ws.title for ws in wb.worksheets if not ws.title.startswith("_")]
        for ws in wb.worksheets:
            if ws.title.startswith("_"):
                continue
            xc.add_back_link(ws, row=1, col=19, target="1 Dashboard")

        wb.active = 0
        path = OUTPUT_DIR / "SOP_Command_Center.xlsx"
        wb.save(path)
        charts = sum(len(ws._charts) for ws in wb.worksheets)
        logger.info("Workbook written: %s (%d sheets, %d charts, %.1f MB).",
                    path.name, len(names), charts, path.stat().st_size / 1e6)
        return path


def build_workbook(config: AppConfig, forecast, consensus, supply, kpi,
                   insights, data=None, validation_report=None,
                   full_master_data: bool = False) -> Path:
    return CommandCenterWorkbook(config).build(
        forecast, consensus, supply, kpi, insights,
        data=data, validation_report=validation_report,
        full_master_data=full_master_data)
