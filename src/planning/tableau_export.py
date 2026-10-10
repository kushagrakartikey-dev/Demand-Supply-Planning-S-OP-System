"""
Tableau export — star schema, tidy facts and a native ``.hyper`` extract.

The Excel workbook and the HTML dashboard are *finished* artefacts: someone
else decided what the charts show. This export is the opposite — it hands over
a clean, correctly-typed data model so an analyst can build whatever they want
in Tableau without cleaning anything first.

Why the report tables are not reused
------------------------------------
Report tables are wide, pre-aggregated and full of presentation artefacts:
merged headers, thousands separators, percentages already multiplied by 100,
mixed grains in one sheet. Every one of those is a defect in Tableau. This
module goes back to the pipeline results and re-shapes them properly.

Modelling decisions
-------------------
* **Star schema.** Four conformed dimensions (SKU, supplier, warehouse, date)
  and seven fact tables at explicit, single grains. Tableau's relationship
  model resolves this natively — no blending, no duplicated measures.
* **Long, not wide.** Accuracy metrics are emitted as
  ``SKU × Model × Stage × Metric × Value`` rather than a column per metric, so
  a single "Metric" filter switches the whole view. Wide tables force one
  worksheet per measure.
* **One Big Table as well.** Purists dislike it, but the fastest path to a
  first chart is a single flat SKU-week table. Both are provided; the star
  schema is there when the flat table stops being enough.
* **Types are explicit.** ISO-8601 dates, real booleans, no separators, no
  stray percent signs. Tableau infers types on first connect and inherits any
  mistake permanently.
* **A ``.hyper`` extract.** One file, columnar, compressed, opens instantly.
  CSVs are also written because they are inspectable and diff-able, and because
  not every reviewer wants a binary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, Keys, OUTPUT_DIR
from src.shared.utils import get_logger

logger = get_logger("tableau_export")

try:
    from tableauhyperapi import (Connection, CreateMode, HyperProcess, Inserter,
                                 NOT_NULLABLE, NULLABLE, SqlType, TableDefinition,
                                 TableName, Telemetry)
    HAS_HYPER = True
except ImportError:                                            # pragma: no cover
    HAS_HYPER = False


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def tidy(df: pd.DataFrame, dates: Optional[List[str]] = None,
         bools: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalise a frame for Tableau ingestion.

    Booleans become real booleans (not "True"/"1"/"yes" strings), dates become
    timezone-naive datetimes, floats are rounded to a sane precision, and
    infinities become nulls. Tableau types a column from its first read and
    keeps that type forever, so cleaning has to happen here, not later.
    """
    out = df.copy()
    for column in (dates or []):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in (bools or []):
        if column in out.columns:
            out[column] = out[column].astype(bool)
    for column in out.select_dtypes(include=["float64", "float32"]).columns:
        out[column] = out[column].replace([np.inf, -np.inf], np.nan).round(4)
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].where(out[column].notna(), None)
    return out


class TableauExporter:
    """Builds the dimensional model, the flat table and the Hyper extract."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = OUTPUT_DIR / "tableau"
        self.data_dir = self.root / "data"
        self.tables: Dict[str, pd.DataFrame] = {}
        self.grains: Dict[str, str] = {}

    def _register(self, name: str, frame: pd.DataFrame, grain: str) -> None:
        if frame is None or frame.empty:
            logger.warning("Tableau table '%s' is empty and was skipped.", name)
            return
        self.tables[name] = frame.reset_index(drop=True)
        self.grains[name] = grain

    # ------------------------------------------------------------------ #
    # Dimensions
    # ------------------------------------------------------------------ #
    def _dimensions(self, data, supply) -> None:
        sku = data.sku_master.merge(
            supply.inventory_policy[[c for c in [Keys.SKU, "ABC_Class", "XYZ_Class",
                                                 "ABC_XYZ", "Service_Level_Target",
                                                 "Safety_Stock_Units",
                                                 "Reorder_Point_Units"]
                                     if c in supply.inventory_policy.columns]],
            on=Keys.SKU, how="left")
        keep = [c for c in [Keys.SKU, "SKU_Name", Keys.CATEGORY, "Sub_Category",
                            Keys.CHANNEL, Keys.SUPPLIER, Keys.WAREHOUSE,
                            "Lifecycle_Status", "Unit_Cost", "Unit_Price",
                            "Shelf_Life_Days", "ABC_Class", "XYZ_Class", "ABC_XYZ",
                            "Service_Level_Target", "Safety_Stock_Units",
                            "Reorder_Point_Units"] if c in sku.columns]
        sku = sku[keep].copy()
        sku["Gross_Margin_Pct"] = (
            (sku["Unit_Price"] - sku["Unit_Cost"]) / sku["Unit_Price"] * 100).round(2)
        self._register("dim_sku", tidy(sku), "one row per SKU")

        supplier = data.supplier_master.copy()
        supplier["Lead_Time_Band"] = pd.cut(
            supplier["Lead_Time_Weeks"], [-0.01, 2, 4, 6, 99],
            labels=["<=2 weeks", "2-4 weeks", "4-6 weeks", ">6 weeks"]).astype(str)
        self._register("dim_supplier",
                       tidy(supplier, bools=["Is_Single_Source"]),
                       "one row per supplier")

        self._register("dim_warehouse", tidy(data.warehouse_master),
                       "one row per warehouse")

        # -- conformed calendar ------------------------------------------- #
        history = data.sales_history[[Keys.WEEK, Keys.WEEK_INDEX, "ISO_Week",
                                      "Month", "Year", "Festival_Intensity",
                                      "Festival_Names", "Is_Festival_Week"]]
        history = history.drop_duplicates(Keys.WEEK).copy()
        history["Period_Type"] = "Actual"
        history["Horizon_Week"] = np.nan

        forecast_weeks = self._forecast_calendar()
        calendar = pd.concat([history, forecast_weeks], ignore_index=True)
        calendar[Keys.WEEK] = pd.to_datetime(calendar[Keys.WEEK])
        calendar = calendar.sort_values(Keys.WEEK).drop_duplicates(Keys.WEEK)
        calendar["Month_Name"] = calendar[Keys.WEEK].dt.strftime("%b %Y")
        calendar["Month_Start"] = calendar[Keys.WEEK].values.astype("datetime64[M]")
        calendar["Quarter"] = ("Q" + calendar[Keys.WEEK].dt.quarter.astype(str)
                               + " " + calendar[Keys.WEEK].dt.year.astype(str))
        calendar["Week_Label"] = ("W" + calendar["ISO_Week"].astype("Int64").astype(str)
                                  + " " + calendar[Keys.WEEK].dt.year.astype(str))
        self._register("dim_date",
                       tidy(calendar, dates=[Keys.WEEK, "Month_Start"],
                            bools=["Is_Festival_Week"]),
                       "one row per week")

    def _forecast_calendar(self) -> pd.DataFrame:
        from src.planning.config import PROCESSED_DIR
        path = PROCESSED_DIR / "future_calendar.csv"
        if not path.exists():
            return pd.DataFrame()
        cal = pd.read_csv(path, parse_dates=[Keys.WEEK])
        weeks = (cal.groupby(Keys.WEEK, as_index=False)
                 .agg(ISO_Week=("ISO_Week", "first"), Month=("Month", "first"),
                      Festival_Intensity=("Festival_Intensity", "first"),
                      Horizon_Week=("horizon", "first")))
        weeks["Year"] = weeks[Keys.WEEK].dt.year
        weeks["Week_Index"] = np.nan
        weeks["Festival_Names"] = ""
        weeks["Is_Festival_Week"] = weeks["Festival_Intensity"] > 0.05
        weeks["Period_Type"] = "Forecast"
        return weeks

    # ------------------------------------------------------------------ #
    # Facts
    # ------------------------------------------------------------------ #
    def _facts(self, data, forecast, consensus, supply, kpi, insights) -> None:
        # 1 -- demand history -------------------------------------------- #
        history = data.sales_history[[
            Keys.SKU, Keys.WEEK, Keys.WEEK_INDEX, "Demand_Units", "Sales_Units",
            "Lost_Sales_Units", "Price", "Is_Promo", "Promo_Depth", "Is_Stockout",
            "Revenue"]].copy()
        history["Fill_Rate_Pct"] = np.where(
            history["Demand_Units"] > 0,
            history["Sales_Units"] / history["Demand_Units"] * 100, 100.0).round(2)
        self._register("fact_demand_history",
                       tidy(history, dates=[Keys.WEEK],
                            bools=["Is_Promo", "Is_Stockout"]),
                       "one row per SKU per historical week")

        # 2 -- forecast --------------------------------------------------- #
        fc = forecast.forecast[[c for c in [
            Keys.SKU, Keys.WEEK, "horizon", "Forecast_Units", "P10", "P90",
            "P2_5", "P97_5", "Forecast_Value", "Selected_Model", "CV_MAPE",
            "Is_Promo", "Festival_Intensity"] if c in forecast.forecast.columns]].copy()
        fc = fc.rename(columns={"horizon": "Horizon_Week"})
        fc["Interval_Width_80"] = (fc["P90"] - fc["P10"]).round(2)
        fc["Uncertainty_Ratio"] = (
            fc["Interval_Width_80"] / fc["Forecast_Units"].replace(0, np.nan)).round(4)
        self._register("fact_forecast",
                       tidy(fc, dates=[Keys.WEEK], bools=["Is_Promo"]),
                       "one row per SKU per forecast week")

        # 3 -- accuracy, in long form ------------------------------------- #
        scores = forecast.model_scores
        if not scores.empty:
            metrics = [c for c in ["MAPE", "sMAPE", "WAPE", "MAE", "RMSE", "Bias",
                                   "Bias_Pct", "Tracking_Signal", "FVA_vs_Naive"]
                       if c in scores.columns]
            ids = [c for c in [Keys.SKU, "Model", "Stage", Keys.CATEGORY,
                               Keys.CHANNEL, "Total_Actual"] if c in scores.columns]
            long = scores.melt(id_vars=ids, value_vars=metrics,
                               var_name="Metric", value_name="Metric_Value")
            long = long[np.isfinite(long["Metric_Value"])]
            # A long metric table means one Metric filter drives every accuracy
            # view, instead of a separate worksheet per measure.
            self._register("fact_accuracy_long", tidy(long),
                           "one row per SKU per model per stage per metric")

        hierarchy = forecast.hierarchy_accuracy
        if not hierarchy.empty:
            self._register("fact_accuracy_by_level", tidy(hierarchy),
                           "one row per aggregation level")

        # 4 -- consensus --------------------------------------------------- #
        cons = consensus.consensus_plan[[c for c in [
            Keys.SKU, Keys.WEEK, "horizon", "Scenario", "Forecast_Units",
            "Commercial_Forecast", "Supply_Ceiling", "Unconstrained_Plan",
            "Consensus_Units", "Constrained_Out_Units", "Consensus_Value",
            "Constrained_Out_Value", "Override_Deviation_Pct", "Commercial_Driver",
            "Override_Alert", "Requires_Board_Signoff", "Is_Constrained",
            "Has_Commercial_Input", "Planner_Decision"]
            if c in consensus.consensus_plan.columns]].copy()
        cons = cons.rename(columns={"horizon": "Horizon_Week"})
        self._register("fact_consensus", tidy(
            cons, dates=[Keys.WEEK],
            bools=["Override_Alert", "Requires_Board_Signoff", "Is_Constrained",
                   "Has_Commercial_Input"]),
            "one row per SKU per week per scenario")

        # 5 -- supply plan -------------------------------------------------- #
        plan = supply.supply_plan[[c for c in [
            Keys.SKU, "horizon", "Opening_Inventory", "Demand_Units",
            "Open_PO_Receipts", "Planned_Receipts", "Gross_Requirement",
            "Net_Requirement", "Planned_Order_Qty", "Closing_Inventory",
            "Safety_Stock", "Weeks_Of_Cover", "Shortage_Units", "Working_Capital",
            "Inventory_Status", "Is_Stockout_Week", "Below_Safety_Stock",
            "Supply_Risk_Index", "Risk_Band", "Lead_Time_Weeks"]
            if c in supply.supply_plan.columns]].copy()
        plan = plan.rename(columns={"horizon": "Horizon_Week"})
        self._register("fact_supply_plan", tidy(
            plan, bools=["Is_Stockout_Week", "Below_Safety_Stock"]),
            "one row per SKU per horizon week")

        risk = supply.sku_summary[[c for c in [
            Keys.SKU, "Supply_Risk_Index", "Risk_Band", "Min_Cover", "Avg_Cover",
            "Total_Demand", "Total_Shortage", "Stockout_Weeks", "Below_SS_Weeks",
            "First_Stockout_Week", "Lead_Time_Weeks", "Can_Recover_In_Lead_Time",
            "Supplier_Peak_Utilisation_Pct", "Recommended_Action",
            "Risk_Factor_Stockout_Proximity", "Risk_Factor_Capacity",
            "Risk_Factor_Cover", "Risk_Factor_Lead_Time",
            "Risk_Factor_Uncertainty", "Avg_Working_Capital"]
            if c in supply.sku_summary.columns]].copy()
        self._register("fact_sku_risk",
                       tidy(risk, bools=["Can_Recover_In_Lead_Time"]),
                       "one row per SKU")

        # 6 -- capacity ------------------------------------------------------ #
        capacity = supply.capacity_alerts.rename(columns={"horizon": "Horizon_Week"})
        self._register("fact_capacity", tidy(capacity),
                       "one row per resource per horizon week")

        # 7 -- orders / OTIF -------------------------------------------------- #
        orders = kpi.order_detail[[c for c in [
            Keys.ORDER, Keys.SKU, Keys.SUPPLIER, Keys.CATEGORY, Keys.CHANNEL,
            Keys.WAREHOUSE, "Promised_Date", "Actual_Delivery_Date", "Order_Month",
            "Ordered_Qty", "Delivered_Qty", "Shortfall_Units", "Fill_Rate",
            "Delay_Days", "On_Time", "In_Full", "OTIF", "Failure_Mode",
            "Root_Cause", "Order_Value"] if c in kpi.order_detail.columns]].copy()
        orders["Fill_Rate_Pct"] = (orders["Fill_Rate"] * 100).round(2)
        orders["OTIF_Flag"] = orders["OTIF"].astype(int)
        # An integer flag alongside the boolean means AVG([OTIF Flag]) gives the
        # OTIF rate directly, which is the single most common Tableau need here.
        self._register("fact_orders", tidy(
            orders, dates=["Promised_Date", "Actual_Delivery_Date"],
            bools=["On_Time", "In_Full", "OTIF"]),
            "one row per delivery order")

        # 8 -- SPC ------------------------------------------------------------ #
        spc = kpi.spc_chart.copy()
        spc["Month_Start"] = pd.to_datetime(spc["Order_Month"] + "-01")
        self._register("fact_spc", tidy(
            spc, dates=["Month_Start"],
            bools=["Special_Cause", "Below_2Sigma_Alert"]),
            "one row per month")

        if not kpi.root_cause_analysis.empty:
            self._register("fact_root_cause",
                           tidy(kpi.root_cause_analysis, bools=["Vital_Few"]),
                           "one row per root cause")

        if insights is not None and not insights.empty:
            self._register("fact_insights", tidy(insights),
                           "one row per generated insight")

    # ------------------------------------------------------------------ #
    # One Big Table
    # ------------------------------------------------------------------ #
    def _flat_table(self) -> None:
        """
        A single denormalised SKU-week table.

        Deliberately redundant with the star schema. The star schema is the
        correct model; this is the one an analyst opens when they want a chart
        in ninety seconds. Offering only the "right" model and expecting people
        to build relationships before their first bar chart is how good data
        models go unused.
        """
        plan = self.tables.get("fact_supply_plan")
        forecast = self.tables.get("fact_forecast")
        sku = self.tables.get("dim_sku")
        if plan is None or sku is None:
            return

        flat = plan.merge(sku, on=Keys.SKU, how="left")
        if forecast is not None:
            cols = [Keys.SKU, "Horizon_Week", "Forecast_Units", "P10", "P90",
                    "Forecast_Value", "Selected_Model", Keys.WEEK]
            flat = flat.merge(forecast[[c for c in cols if c in forecast.columns]],
                              on=[Keys.SKU, "Horizon_Week"], how="left")

        supplier = self.tables.get("dim_supplier")
        if supplier is not None:
            keep = [Keys.SUPPLIER, "Supplier_Name", "Archetype", "Country",
                    "Reliability_Score", "Lead_Time_Band", "Is_Single_Source"]
            flat = flat.merge(supplier[[c for c in keep if c in supplier.columns]],
                              on=Keys.SUPPLIER, how="left")

        risk = self.tables.get("fact_sku_risk")
        if risk is not None:
            keep = [Keys.SKU, "Can_Recover_In_Lead_Time", "First_Stockout_Week",
                    "Recommended_Action", "Min_Cover"]
            flat = flat.merge(risk[[c for c in keep if c in risk.columns]],
                              on=Keys.SKU, how="left")

        flat["Revenue_At_Risk"] = (flat["Shortage_Units"].fillna(0)
                                   * flat["Unit_Price"].fillna(0)).round(2)
        flat["Inventory_Value"] = (flat["Closing_Inventory"].clip(lower=0)
                                   * flat["Unit_Cost"].fillna(0)).round(2)
        self._register("flat_sku_week", tidy(flat),
                       "one row per SKU per horizon week, fully denormalised")

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def _write_csvs(self) -> List[Path]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, frame in self.tables.items():
            path = self.data_dir / f"{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d",
                         encoding="utf-8-sig")
            written.append(path)
        return written

    @staticmethod
    def _sql_type(series: pd.Series):
        if pd.api.types.is_bool_dtype(series):
            return SqlType.bool()
        if pd.api.types.is_integer_dtype(series):
            return SqlType.big_int()
        if pd.api.types.is_float_dtype(series):
            return SqlType.double()
        if pd.api.types.is_datetime64_any_dtype(series):
            return SqlType.date()
        return SqlType.text()

    def _write_hyper(self) -> Optional[Path]:
        """Write every table into one multi-table ``.hyper`` extract."""
        if not HAS_HYPER:
            logger.warning("tableauhyperapi unavailable — CSV output only.")
            return None
        path = self.root / "SOP_Analytics.hyper"
        try:
            with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU
                              ) as hyper:
                with Connection(hyper.endpoint, str(path),
                                CreateMode.CREATE_AND_REPLACE) as conn:
                    conn.catalog.create_schema("Extract")
                    for name, frame in self.tables.items():
                        columns = [
                            TableDefinition.Column(str(c), self._sql_type(frame[c]),
                                                   NULLABLE)
                            for c in frame.columns]
                        table = TableDefinition(TableName("Extract", name), columns)
                        conn.catalog.create_table(table)

                        prepared = frame.copy()
                        for column in prepared.columns:
                            series = prepared[column]
                            if pd.api.types.is_datetime64_any_dtype(series):
                                prepared[column] = series.dt.date
                            elif pd.api.types.is_bool_dtype(series):
                                prepared[column] = series.astype(bool)
                            elif pd.api.types.is_integer_dtype(series):
                                prepared[column] = series.astype("Int64")
                        rows = prepared.astype(object).where(
                            pd.notna(prepared), None).values.tolist()
                        with Inserter(conn, table) as inserter:
                            inserter.add_rows(rows)
                            inserter.execute()
            logger.info("Hyper extract written: %s (%.1f MB, %d tables).",
                        path.name, path.stat().st_size / 1e6, len(self.tables))
            return path
        except Exception as exc:                               # pragma: no cover
            logger.error("Hyper extract failed (%s); CSVs are unaffected.", exc)
            return None

    def _write_dictionary(self) -> Path:
        """Field dictionary plus suggested starting views."""
        rows = []
        for name, frame in self.tables.items():
            for column in frame.columns:
                series = frame[column]
                if pd.api.types.is_bool_dtype(series):
                    role, dtype = "Dimension", "Boolean"
                elif pd.api.types.is_numeric_dtype(series):
                    key = column.endswith("_ID") or column in ("Horizon_Week",
                                                               "Week_Index")
                    role = "Dimension" if key else "Measure"
                    dtype = "Number"
                elif pd.api.types.is_datetime64_any_dtype(series):
                    role, dtype = "Dimension", "Date"
                else:
                    role, dtype = "Dimension", "Text"
                rows.append({
                    "Table": name, "Field": column, "Suggested_Role": role,
                    "Data_Type": dtype,
                    "Distinct_Values": int(series.nunique(dropna=True)),
                    "Null_Pct": round(series.isna().mean() * 100, 2),
                })
        dictionary = pd.DataFrame(rows)
        path = self.root / "field_dictionary.csv"
        dictionary.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _write_readme(self, hyper: Optional[Path]) -> Path:
        lines = [
            "# Tableau Data Pack — S&OP Intelligence System",
            "",
            "> All data is synthetic and simulated for portfolio demonstration.",
            "> It does not originate from any employer, client or real organisation.",
            "",
            f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}",
            "",
            "## What is here",
            "",
            "| File | Purpose |",
            "|---|---|",
        ]
        if hyper:
            lines.append("| `SOP_Analytics.hyper` | **Start here.** Multi-table native "
                         "extract — open directly in Tableau Desktop/Public. |")
        lines += [
            "| `data/*.csv` | The same tables as CSV, for inspection or other tools. |",
            "| `field_dictionary.csv` | Every field with suggested Dimension/Measure "
            "role, type, cardinality and null rate. |",
            "",
            "## Data model",
            "",
            "Star schema. Relate the facts to the dimensions on the keys below "
            "(Tableau relationships, not joins — this keeps grains independent and "
            "avoids measure duplication).",
            "",
            "| Table | Grain | Relate to |",
            "|---|---|---|",
        ]
        for name, frame in self.tables.items():
            relations = []
            for key, dim in ((Keys.SKU, "dim_sku"), (Keys.SUPPLIER, "dim_supplier"),
                             (Keys.WAREHOUSE, "dim_warehouse"), (Keys.WEEK, "dim_date")):
                if key in frame.columns and not name.startswith("dim_"):
                    relations.append(f"`{dim}` on `{key}`")
            lines.append(f"| `{name}` | {self.grains[name]} | "
                         f"{'; '.join(relations) if relations else '—'} |")

        lines += [
            "",
            "## Two ways in",
            "",
            "* **`flat_sku_week`** — one denormalised SKU-week table. Drag and drop, "
            "no relationships needed. Use this for a first chart.",
            "* **Star schema** — correct model, no measure duplication across grains. "
            "Use this once the flat table stops being enough.",
            "",
            "## Suggested starting views",
            "",
            "| View | Table | Build |",
            "|---|---|---|",
            "| Forecast accuracy ladder | `fact_accuracy_by_level` | "
            "Bar: `Aggregation_Level` × `WAPE`, reference line at 15 |",
            "| Model competition | `fact_accuracy_long` | "
            "Bar: `Model` × AVG(`Metric_Value`), filter `Metric` = MAPE, "
            "`Stage` = HOLDOUT |",
            "| OTIF control chart | `fact_spc` | Line: `Month_Start` × `OTIF_Pct` "
            "plus `Center_Line`, `UCL`, `LCL`; colour by `Special_Cause` |",
            "| OTIF by any dimension | `fact_orders` | AVG(`OTIF_Flag`) × 100 on any "
            "dimension — the integer flag makes this a one-click measure |",
            "| Supplier capacity heatmap | `fact_capacity` | "
            "`Resource_ID` × `Horizon_Week`, colour by `Utilisation_Pct` |",
            "| Inventory trajectory | `fact_supply_plan` | "
            "`Horizon_Week` × SUM(`Closing_Inventory`) vs SUM(`Safety_Stock`) |",
            "| Risk quadrant | `fact_sku_risk` | Scatter `Min_Cover` × "
            "`Supply_Risk_Index`, colour `Risk_Band`, size `Total_Demand` |",
            "| Consensus bridge | `fact_consensus` | "
            "SUM of `Forecast_Units`, `Commercial_Forecast`, `Consensus_Units` "
            "by `Horizon_Week`, filter `Scenario` |",
            "| Root-cause Pareto | `fact_root_cause` | Bar `Value_At_Risk` + line "
            "`Cumulative_Value_Pct` on dual axis |",
            "| Demand vs promo | `fact_demand_history` | "
            "`Week_Start` × SUM(`Demand_Units`), colour by `Is_Promo` |",
            "",
            "## Calculated fields worth creating",
            "",
            "```",
            "OTIF %            = AVG([OTIF Flag]) * 100",
            "Forecast Error %  = ABS(SUM([Forecast Units]) - SUM([Demand Units]))",
            "                    / SUM([Demand Units]) * 100",
            "Cover Status      = IF [Weeks Of Cover] < 1.5 THEN 'Critical'",
            "                    ELSEIF [Weeks Of Cover] < 4 THEN 'Below target'",
            "                    ELSEIF [Weeks Of Cover] > 8 THEN 'Excess'",
            "                    ELSE 'Healthy' END",
            "Revenue At Risk   = SUM([Shortage Units]) * AVG([Unit Price])",
            "```",
            "",
            "## Notes on the modelling",
            "",
            "* `fact_accuracy_long` is deliberately long (`Metric` / `Metric_Value`) "
            "so one filter drives every accuracy view instead of a worksheet per "
            "measure.",
            "* `fact_orders` carries `OTIF_Flag` as an integer beside the boolean, "
            "because `AVG()` of a boolean is awkward in Tableau.",
            "* `dim_date` spans both history and the forecast horizon, separated by "
            "`Period_Type` — so actuals and forecast can sit on one continuous axis.",
            "* Percentages are stored as numbers already multiplied by 100 "
            "(`87.4`, not `0.874`). Format as number, not percent.",
        ]
        path = self.root / "README.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ #
    def run(self, data, forecast, consensus, supply, kpi,
            insights: pd.DataFrame) -> Dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        self._dimensions(data, supply)
        self._facts(data, forecast, consensus, supply, kpi, insights)
        self._flat_table()

        csvs = self._write_csvs()
        hyper = self._write_hyper()
        dictionary = self._write_dictionary()
        readme = self._write_readme(hyper)

        total_rows = sum(len(f) for f in self.tables.values())
        logger.info("Tableau pack: %d tables, %s rows, %d CSVs%s.",
                    len(self.tables), f"{total_rows:,}", len(csvs),
                    ", 1 hyper extract" if hyper else "")
        return {"hyper": hyper, "csvs": csvs, "dictionary": dictionary,
                "readme": readme, "tables": len(self.tables),
                "rows": total_rows, "dir": self.root}


def export_for_tableau(config: AppConfig, data, forecast, consensus, supply,
                       kpi, insights) -> Dict[str, object]:
    return TableauExporter(config).run(data, forecast, consensus, supply,
                                       kpi, insights)
