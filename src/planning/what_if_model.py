"""
Interactive what-if model — a genuinely live Excel calculation sheet.

Every result on this sheet is computed by **Excel**, not by Python. Change a
service level, a lead time or a demand assumption and the safety stock, reorder
point, EOQ, inventory value and total annual cost all recalculate in the
spreadsheet with no pipeline re-run.

Why this sheet exists
---------------------
The rest of the workbook is a read-and-decide artefact: Python is the
calculation engine and Excel is the presentation layer. That is the right split
for anything involving rolling-origin validation or a gradient-boosted model,
which cannot be honestly expressed in cell formulas.

But inventory policy *can* be, and a planner's most common question — "what
happens to my buffer if I move this A-item from 95% to 99%?" — should not
require anyone to run Python. This sheet answers it in the tool they already
have open.

Function compatibility
----------------------
Only functions in the ECMA-376 core set are used (``NORMSINV``, ``INDEX``,
``MATCH``, ``SQRT``, ``IFERROR``, ``MAX``, ``ROUND``). Functions introduced in
Excel 2010 or later — ``NORM.S.INV``, ``XLOOKUP``, ``IFS`` — require an
``_xlfn.`` prefix when written by a non-Excel library and display as ``#NAME?``
without it. ``NORMSINV`` is the compatibility-era equivalent and works in every
Excel version, LibreOffice and Google Sheets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from src.planning import excel_charts as xc
from src.planning.config import Keys

INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")     # editable cells
RESULT_FILL = PatternFill("solid", fgColor="E8F0FA")    # formula cells
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


class WhatIfModelBuilder:
    """Builds the live inventory what-if sheet."""

    def __init__(self, master, sheet_name: str = "_policy_data") -> None:
        self.master = master
        self.POLICY_SHEET = sheet_name
        self.styler = getattr(master, "styler", None) or master.theme
        self.cfg = master.cfg
        self.config = master.config
        self.currency = getattr(master, "currency", None) or \
            master.cfg.currency_symbol

    # ------------------------------------------------------------------ #
    def _policy_ref(self, column: str, layout: Dict[str, object]) -> Optional[str]:
        """Absolute range for one column of the inventory policy table."""
        columns: List[str] = layout["columns"]          # type: ignore[assignment]
        if column not in columns:
            return None
        letter = get_column_letter(columns.index(column) + 1)
        return (f"'{self.POLICY_SHEET}'!${letter}${layout['first_row']}"
                f":${letter}${layout['last_row']}")

    def _lookup(self, column: str, layout: Dict[str, object], sku_cell: str,
                default: str = "0") -> str:
        """INDEX/MATCH pulling one field for the selected SKU."""
        value_range = self._policy_ref(column, layout)
        key_range = self._policy_ref(Keys.SKU, layout)
        if not value_range or not key_range:
            return f"={default}"
        return (f"=IFERROR(INDEX({value_range},"
                f"MATCH(${sku_cell},{key_range},0)),{default})")

    # ------------------------------------------------------------------ #
    def build(self, ws: Worksheet, wb, layout: Dict[str, object]) -> None:
        s = self.styler
        money = f'"{self.currency}"#,##0'
        supply_cfg = self.config.supply

        ws.column_dimensions["A"].width = 38
        for col in "BCD":
            ws.column_dimensions[col].width = 18
        ws.column_dimensions["E"].width = 52

        row = s.title(
            ws, "INTERACTIVE WHAT-IF MODEL",
            "What if I change this assumption? Every figure below is calculated "
            "by EXCEL, not by Python — edit any yellow cell and watch it move.")

        # -- SKU selector --------------------------------------------------- #
        row = s.band(ws, row, "1 · SELECT A SKU", width=6)
        sku_range = self._policy_ref(Keys.SKU, layout)
        if sku_range:
            try:
                wb.defined_names.add(
                    DefinedName("SKU_LIST", attr_text=sku_range.lstrip("=")))
            except Exception:
                try:
                    wb.defined_names["SKU_LIST"] = DefinedName(
                        "SKU_LIST", attr_text=sku_range.lstrip("="))
                except Exception:
                    pass

        label = ws.cell(row=row, column=1, value="SKU (dropdown)")
        label.font = Font(bold=True, size=11)
        sku_cell_ref = f"B{row}"
        sku_cell = ws.cell(row=row, column=2, value="SKU_0001")
        sku_cell.font = Font(bold=True, size=12, color="FFFFFF")
        sku_cell.fill = PatternFill("solid", fgColor=self.cfg.brand_navy)
        sku_cell.alignment = Alignment(horizontal="center")
        sku_cell.border = BORDER
        if sku_range:
            validation = DataValidation(
                type="list", formula1="=SKU_LIST", allow_blank=False,
                showDropDown=False, promptTitle="Select SKU",
                prompt="Pick any SKU. Its live parameters load from "
                       "sheet '33 Inventory Policy'.")
            ws.add_data_validation(validation)
            validation.add(sku_cell)

        for offset, (col, field) in enumerate(
                [(3, Keys.CATEGORY), (4, "ABC_Class")], start=0):
            formula = self._lookup(field, layout, sku_cell_ref, default='""')
            cell = ws.cell(row=row, column=3 + offset, value=formula)
            cell.font = Font(size=10, italic=True, color="595959")
        row += 2

        # -- Parameters ------------------------------------------------------ #
        row = s.band(ws, row, "2 · PARAMETERS — EDIT THE YELLOW COLUMN", width=6)
        for offset, heading in enumerate(
                ["Parameter", "Baseline (live from data)", "What-If (editable)",
                 "", "Source / note"]):
            cell = ws.cell(row=row, column=1 + offset, value=heading)
            cell.font = s.header_font
            cell.fill = s.header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = BORDER
        row += 1

        params: List[Tuple[str, str, str, str, str]] = [
            ("Average weekly demand", "Avg_Weekly_Forecast", "#,##0.0",
             "0", "12-week forecast mean for this SKU"),
            ("Demand std deviation (weekly)", "Demand_Std_Weekly", "#,##0.0",
             "0", "Forecast ERROR std, not historical demand std"),
            ("Supplier lead time (weeks)", "Lead_Time_Weeks", "0.00",
             "2", "From supplier master"),
            ("Lead time std deviation (weeks)", "Lead_Time_Std_Weeks", "0.00",
             "0", "Supplier reliability variance"),
            ("Target service level", "Service_Level_Target", "0.0%",
             "0.95", "Set by ABC class: A 99%, B 95%, C 90%"),
            ("Unit cost", "Unit_Cost", money, "0", "Standard cost"),
            ("Unit price", "Unit_Price", money, "0", "List selling price"),
            ("Supplier MOQ (units)", "MOQ", "#,##0", "0",
             "Minimum order quantity"),
        ]

        param_rows: Dict[str, int] = {}
        for label_text, field, fmt, default, note in params:
            ws.cell(row=row, column=1, value=label_text).font = s.body
            base = ws.cell(row=row, column=2,
                           value=self._lookup(field, layout, sku_cell_ref, default))
            base.number_format = fmt
            base.fill = RESULT_FILL
            base.border = BORDER
            base.font = s.body

            what_if = ws.cell(row=row, column=3, value=f"=B{row}")
            what_if.number_format = fmt
            what_if.fill = INPUT_FILL
            what_if.border = BORDER
            what_if.font = Font(size=10, bold=True)

            ws.cell(row=row, column=5, value=note).font = Font(size=9, color="595959")
            param_rows[field] = row
            row += 1

        # Assumptions that are policy settings rather than SKU attributes.
        for label_text, key, value, fmt, note in [
            ("Ordering cost per PO", "ordering_cost", supply_cfg.ordering_cost,
             money, "Configured in config.py"),
            ("Annual holding cost rate", "holding_rate", supply_cfg.holding_cost_rate,
             "0.0%", "Annual, as % of unit cost"),
            ("Demand uplift factor", "uplift", 1.0, "0.00",
             "1.00 = no change. Try 1.20 for a +20% demand scenario."),
        ]:
            ws.cell(row=row, column=1, value=label_text).font = s.body
            base = ws.cell(row=row, column=2, value=value)
            base.number_format = fmt
            base.fill = RESULT_FILL
            base.border = BORDER
            what_if = ws.cell(row=row, column=3, value=f"=B{row}")
            what_if.number_format = fmt
            what_if.fill = INPUT_FILL
            what_if.border = BORDER
            what_if.font = Font(size=10, bold=True)
            ws.cell(row=row, column=5, value=note).font = Font(size=9, color="595959")
            param_rows[key] = row
            row += 1

        note = ws.cell(row=row, column=1,
                       value="Yellow cells are editable. Each starts as '=B<row>' so it "
                             "tracks the baseline until you overwrite it. To reset a "
                             "cell, retype the formula shown in column B's row.")
        note.font = Font(italic=True, size=9, color="595959")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=5)
        row += 2

        # -- Results ---------------------------------------------------------- #
        row = s.band(ws, row, "3 · CALCULATED RESULTS — LIVE EXCEL FORMULAS", width=6)
        for offset, heading in enumerate(
                ["Result", "Baseline", "What-If", "Change", "Formula used"]):
            cell = ws.cell(row=row, column=1 + offset, value=heading)
            cell.font = s.header_font
            cell.fill = s.header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = BORDER
        row += 1

        def p(field: str, col: str) -> str:
            """Reference a parameter cell in the given column (B or C)."""
            return f"{col}{param_rows[field]}"

        result_rows: Dict[str, int] = {}
        first_result_row = row

        def add_result(label_text: str, builder, fmt: str, explanation: str,
                       key: str, show_delta: bool = True) -> None:
            nonlocal row
            ws.cell(row=row, column=1, value=label_text).font = s.body
            for column_letter, index in (("B", 2), ("C", 3)):
                cell = ws.cell(row=row, column=index, value=builder(column_letter))
                cell.number_format = fmt
                cell.fill = RESULT_FILL
                cell.border = BORDER
                cell.font = Font(size=10, bold=(index == 3))
            if show_delta:
                delta = ws.cell(row=row, column=4, value=f"=C{row}-B{row}")
                delta.number_format = fmt
                delta.border = BORDER
                delta.font = Font(size=10, color=self.cfg.brand_navy)
            # A leading '=' would be parsed by Excel as a formula and render
            # as #NAME?. The explanation is documentation, so it is written as
            # plain text.
            ws.cell(row=row, column=5, value=explanation).font = Font(
                size=9, color="595959", name="Consolas")
            result_rows[key] = row
            row += 1

        # Effective demand after the uplift factor.
        add_result("Effective weekly demand", 
                   lambda c: f"={p('Avg_Weekly_Forecast', c)}*{p('uplift', c)}",
                   "#,##0.0", "avg weekly demand x uplift factor", "mu")

        add_result("Safety factor Z",
                   lambda c: f"=NORMSINV({p('Service_Level_Target', c)})",
                   "0.000", "NORMSINV(service level)", "z")

        add_result("Safety stock (units)",
                   lambda c: (f"={c}{result_rows['z']}*SQRT("
                              f"{p('Lead_Time_Weeks', c)}*{p('Demand_Std_Weekly', c)}^2"
                              f"+{c}{result_rows['mu']}^2*"
                              f"{p('Lead_Time_Std_Weeks', c)}^2)"),
                   "#,##0", "Z x SQRT(LT x sigmaD^2 + muD^2 x sigmaLT^2)", "ss")

        add_result("Reorder point (units)",
                   lambda c: (f"={c}{result_rows['mu']}*{p('Lead_Time_Weeks', c)}"
                              f"+{c}{result_rows['ss']}"),
                   "#,##0", "muD x LT + safety stock", "rop")

        add_result("Economic order quantity",
                   lambda c: (f"=IFERROR(SQRT(2*{c}{result_rows['mu']}*52*"
                              f"{p('ordering_cost', c)}/({p('holding_rate', c)}*"
                              f"{p('Unit_Cost', c)})),0)"),
                   "#,##0", "SQRT(2 x D x S / (H x unit cost))", "eoq")

        add_result("Order quantity (MOQ applied)",
                   lambda c: f"=MAX({c}{result_rows['eoq']},{p('MOQ', c)})",
                   "#,##0", "MAX(EOQ, supplier MOQ)", "q")

        add_result("Orders per year",
                   lambda c: (f"=IFERROR({c}{result_rows['mu']}*52/"
                              f"{c}{result_rows['q']},0)"),
                   "0.0", "annual demand / order quantity", "orders")

        add_result("Average inventory (units)",
                   lambda c: f"={c}{result_rows['ss']}+{c}{result_rows['q']}/2",
                   "#,##0", "safety stock + cycle stock (Q/2)", "avginv")

        add_result("Safety stock value",
                   lambda c: f"={c}{result_rows['ss']}*{p('Unit_Cost', c)}",
                   money, "safety stock x unit cost", "ssval")

        add_result("Total inventory value",
                   lambda c: f"={c}{result_rows['avginv']}*{p('Unit_Cost', c)}",
                   money, "average inventory x unit cost", "invval")

        add_result("Annual holding cost",
                   lambda c: (f"={c}{result_rows['avginv']}*{p('Unit_Cost', c)}"
                              f"*{p('holding_rate', c)}"),
                   money, "avg inventory x unit cost x holding rate", "hold")

        add_result("Annual ordering cost",
                   lambda c: f"={c}{result_rows['orders']}*{p('ordering_cost', c)}",
                   money, "orders per year x cost per PO", "order")

        add_result("Total annual inventory cost",
                   lambda c: f"={c}{result_rows['hold']}+{c}{result_rows['order']}",
                   money, "holding cost + ordering cost", "total")

        add_result("Weeks of cover (safety stock)",
                   lambda c: (f"=IFERROR({c}{result_rows['ss']}/"
                              f"{c}{result_rows['mu']},0)"),
                   "0.00", "safety stock / weekly demand", "cover")

        add_result("Expected stockout cycles / year",
                   lambda c: (f"={c}{result_rows['orders']}*"
                              f"(1-{p('Service_Level_Target', c)})"),
                   "0.00", "orders/yr x (1 - cycle service level)", "stockouts")

        add_result("Annual revenue exposure",
                   lambda c: (f"={c}{result_rows['stockouts']}*"
                              f"{c}{result_rows['mu']}*{p('Lead_Time_Weeks', c)}"
                              f"*{p('Unit_Price', c)}"),
                   money, "stockout cycles x demand during LT x unit price", "exposure")

        row += 1

        # -- Sensitivity ------------------------------------------------------ #
        row = s.band(ws, row, "4 · SERVICE LEVEL SENSITIVITY — THE COST OF SERVICE",
                     width=6)
        note = ws.cell(
            row=row, column=1,
            value="Uses the What-If column parameters. Safety stock rises "
                  "non-linearly with service level: the last few percent are by far "
                  "the most expensive, which is why service targets are set by ABC "
                  "class rather than uniformly.")
        note.font = Font(italic=True, size=9, color="595959")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=5)
        row += 2

        for offset, heading in enumerate(
                ["Service level", "Z score", "Safety stock (units)",
                 "Safety stock value", "Extra cost vs 90%"]):
            cell = ws.cell(row=row, column=1 + offset, value=heading)
            cell.font = s.header_font
            cell.fill = s.header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = BORDER
        sens_header = row
        row += 1

        levels = [0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
        sens_first = row
        base_value_row = sens_first + 1          # the 90% row is the comparator
        for level in levels:
            level_cell = ws.cell(row=row, column=1, value=level)
            level_cell.number_format = "0.0%"
            level_cell.fill = INPUT_FILL
            level_cell.border = BORDER

            z = ws.cell(row=row, column=2, value=f"=NORMSINV(A{row})")
            z.number_format = "0.000"

            ss = ws.cell(row=row, column=3, value=(
                f"=B{row}*SQRT({p('Lead_Time_Weeks', 'C')}*"
                f"{p('Demand_Std_Weekly', 'C')}^2+C{result_rows['mu']}^2*"
                f"{p('Lead_Time_Std_Weeks', 'C')}^2)"))
            ss.number_format = "#,##0"

            value = ws.cell(row=row, column=4,
                            value=f"=C{row}*{p('Unit_Cost', 'C')}")
            value.number_format = money

            extra = ws.cell(row=row, column=5, value=f"=D{row}-$D${base_value_row}")
            extra.number_format = money

            for col in range(2, 6):
                ws.cell(row=row, column=col).fill = RESULT_FILL
                ws.cell(row=row, column=col).border = BORDER
                ws.cell(row=row, column=col).font = s.body
            row += 1
        sens_last = row - 1

        # Live chart bound to the sensitivity block.
        chart = LineChart()
        chart.title = "Safety Stock Value vs Service Level (live)"
        chart.height, chart.width = 9, 20
        chart.y_axis.title = f"Safety stock value ({self.currency})"
        chart.x_axis.title = "Cycle service level"
        chart.x_axis.delete = False
        chart.y_axis.delete = False
        data = Reference(ws, min_col=4, min_row=sens_header, max_row=sens_last)
        categories = Reference(ws, min_col=1, min_row=sens_first, max_row=sens_last)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        if chart.series:
            chart.series[0].graphicalProperties.line.solidFill = self.cfg.brand_navy
            chart.series[0].graphicalProperties.line.width = 26_000
        xc.place_chart(ws, chart, f"A{row + 1}")
        row += 20

        # -- Notes ------------------------------------------------------------- #
        row = s.band(ws, row, "5 · HOW THIS SHEET WORKS", width=6)
        notes = pd.DataFrame([
            {"Point": "Excel does the maths here",
             "Detail": "Every value in sections 3 and 4 is a live formula. Nothing on "
                       "this sheet was computed in Python. Click any cell to inspect it."},
            {"Point": "Formulas match the pipeline",
             "Detail": "The safety-stock, EOQ and reorder-point formulas are identical "
                       "to those in src/shared/metrics.py and METHODOLOGY.md. Cross-"
                       "check the baseline column against sheet 33."},
            {"Point": "Why sigma_D is forecast error",
             "Detail": "The buffer absorbs error in the plan, not total demand "
                       "variation. Sizing on historical demand std over-buffers "
                       "predictable SKUs and under-buffers volatile ones."},
            {"Point": "Why lead-time variance matters",
             "Detail": "Dropping the muD^2 x sigmaLT^2 term is the usual spreadsheet "
                       "shortcut. It under-buffers exactly the unreliable suppliers "
                       "that need the buffer most. Set lead time std to 0 to see it."},
            {"Point": "Service level is non-linear",
             "Detail": "Section 4 shows why. Moving 99% to 99.9% typically costs more "
                       "buffer than moving 80% to 95% did."},
            {"Point": "Compatibility",
             "Detail": "NORMSINV is used rather than NORM.S.INV so the sheet works in "
                       "every Excel version, LibreOffice and Google Sheets."},
            {"Point": "This is a single-SKU model",
             "Detail": "Portfolio-wide optimisation across 500 SKUs is the pipeline's "
                       "job (sheet 33). This sheet is for interrogating one SKU."},
        ])
        s.table(ws, notes, row, max_rows=10)

        xc.add_back_link(ws)
        ws.sheet_view.showGridLines = False
        xc.setup_print(ws, landscape=False, title="What-If Model")
