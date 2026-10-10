"""
Module 6 -- Interactive S&OP Command Dashboard.

A four-tab Plotly Dash application over the pipeline's published datasets.

Design principles
-----------------
* **Reads, never computes.** Every figure is rendered from a CSV written by the
  pipeline. The dashboard cannot invent a KPI, and a number shown here is
  provably the same number that appears in the Excel pack, because both read
  the same file. This is the single most important property of a decision-
  support tool: one version of the truth.
* **Degrades honestly.** If a dataset is missing, the affected panel says so
  instead of rendering an empty axis that reads as "zero".
* **Exception-first.** Default views sort to the problems. A planner opening
  this at 8am should see what is broken, not a tidy alphabetical list.

Run with::

    python dashboard/dashboard.py
    python run_demo.py --open-dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
from dash import Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go
import plotly.express as px

from src.planning.config import PROCESSED_DIR, Keys, load_config
from src.shared.utils import get_logger, human_number

logger = get_logger("dashboard")
CONFIG = load_config()
RC = CONFIG.report

# --------------------------------------------------------------------------- #
# Visual language
# --------------------------------------------------------------------------- #
NAVY = f"#{RC.brand_navy}"
BLUE = f"#{RC.brand_blue}"
LIGHT = f"#{RC.brand_light}"
GREEN = f"#{RC.brand_green}"
AMBER = f"#{RC.brand_amber}"
RED = f"#{RC.brand_red}"
INK = "#1B1F2A"
MUTED = "#6B7280"
SURFACE = "#FFFFFF"
CANVAS = "#F4F6FA"

RAG = {"Green": GREEN, "Amber": AMBER, "Red": RED,
       "World Class": GREEN, "Acceptable": AMBER, "At Risk": RED,
       "OK": GREEN, "WARNING": AMBER, "BREACH": RED,
       "Critical": RED, "High": "#C2410C", "Medium": AMBER, "Info": BLUE}

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
              size=12, color=INK),
    margin=dict(l=55, r=25, t=55, b=45),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
class DashboardData:
    """
    Lazy, cached loader over the pipeline's processed CSV exports.

    A missing file returns an empty frame rather than raising, so one absent
    dataset degrades a single panel instead of taking the whole app down.
    """

    FILES = {
        "forecast": "demand_forecast.csv",
        "hierarchy": "forecast_accuracy_hierarchy.csv",
        "sku_accuracy": "forecast_sku_accuracy.csv",
        "leaderboard": "model_leaderboard.csv",
        "model_scores": "model_scores.csv",
        "weights": "ensemble_weights.csv",
        "importance": "feature_importance.csv",
        "shap": "shap_summary.csv",
        "holdout": "holdout_predictions.csv",
        "consensus": "consensus_plan.csv",
        "override_log": "override_log.csv",
        "scenarios": "scenario_summary.csv",
        "waterfall": "consensus_waterfall.csv",
        "supply_plan": "supply_plan.csv",
        "supply_summary": "supply_sku_summary.csv",
        "capacity": "capacity_alerts.csv",
        "stockouts": "stockout_risks.csv",
        "policy": "inventory_policy.csv",
        "otif_supplier": "otif_by_supplier.csv",
        "otif_category": "otif_by_category.csv",
        "otif_channel": "otif_by_channel.csv",
        "otif_month": "otif_by_month.csv",
        "spc": "spc_chart.csv",
        "spc_violations": "spc_violations.csv",
        "root_cause": "root_cause_analysis.csv",
        "scorecard": "supplier_scorecard.csv",
        "insights": "executive_insights.csv",
    }

    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}

    def get(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            path = PROCESSED_DIR / self.FILES.get(key, f"{key}.csv")
            try:
                self._cache[key] = pd.read_csv(path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                logger.warning("Dataset '%s' not found at %s.", key, path)
                self._cache[key] = pd.DataFrame()
        return self._cache[key]

    def available(self) -> bool:
        return not self.get("forecast").empty


DATA = DashboardData()


# --------------------------------------------------------------------------- #
# Reusable components
# --------------------------------------------------------------------------- #
def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=MUTED))
    fig.update_layout(**PLOT_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def kpi_card(label: str, value: str, sub: str = "", status: str = "neutral") -> html.Div:
    colour = {"good": GREEN, "warn": AMBER, "bad": RED}.get(status, NAVY)
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": MUTED,
                               "textTransform": "uppercase", "letterSpacing": "0.06em",
                               "fontWeight": 600}),
        html.Div(value, style={"fontSize": "26px", "fontWeight": 700,
                               "color": colour, "marginTop": "4px",
                               "lineHeight": "1.1"}),
        html.Div(sub, style={"fontSize": "11px", "color": MUTED, "marginTop": "2px"}),
    ], style={"backgroundColor": SURFACE, "padding": "14px 16px",
              "borderRadius": "10px", "flex": "1", "minWidth": "150px",
              "borderTop": f"3px solid {colour}",
              "boxShadow": "0 1px 3px rgba(16,24,40,0.08)"})


def panel(title: str, children, subtitle: str = "") -> html.Div:
    header = [html.Div(title, style={"fontSize": "14px", "fontWeight": 700,
                                     "color": NAVY})]
    if subtitle:
        header.append(html.Div(subtitle, style={"fontSize": "11px", "color": MUTED,
                                                "marginTop": "2px"}))
    return html.Div([
        html.Div(header, style={"marginBottom": "10px"}),
        children,
    ], style={"backgroundColor": SURFACE, "padding": "16px", "borderRadius": "10px",
              "boxShadow": "0 1px 3px rgba(16,24,40,0.08)", "marginBottom": "16px"})


def data_table(df: pd.DataFrame, table_id: str, page_size: int = 10,
               highlight: Optional[Dict[str, Dict[str, str]]] = None) -> dash_table.DataTable:
    """Standard table with export enabled and RAG conditional formatting."""
    if df is None or df.empty:
        return dash_table.DataTable(
            id=table_id, columns=[{"name": "Status", "id": "Status"}],
            data=[{"Status": "No records available."}])

    frame = df.copy()
    for col in frame.select_dtypes("float").columns:
        frame[col] = frame[col].round(2)

    conditional: List[Dict[str, object]] = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#F8FAFC"}]
    if highlight:
        for column, mapping in highlight.items():
            if column not in frame.columns:
                continue
            for value, colour in mapping.items():
                conditional.append({
                    "if": {"filter_query": f'{{{column}}} = "{value}"',
                           "column_id": column},
                    "backgroundColor": colour, "color": "white", "fontWeight": "600"})

    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c.replace("_", " "), "id": c} for c in frame.columns],
        data=frame.to_dict("records"),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "8px",
                    "fontFamily": "Inter, Segoe UI, sans-serif",
                    "textAlign": "left", "maxWidth": "300px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"backgroundColor": NAVY, "color": "white",
                      "fontWeight": "600", "fontSize": "11px",
                      "textTransform": "uppercase"},
        style_data_conditional=conditional,
    )


def dropdown(component_id: str, options: List[str], value=None, multi: bool = False,
             placeholder: str = "All"):
    return dcc.Dropdown(
        id=component_id,
        options=[{"label": str(o), "value": o} for o in options],
        value=value, multi=multi, placeholder=placeholder, clearable=not multi,
        style={"fontSize": "12px"})


def control(label: str, component) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "11px", "fontWeight": 600,
                                 "color": MUTED, "display": "block",
                                 "marginBottom": "4px",
                                 "textTransform": "uppercase",
                                 "letterSpacing": "0.04em"}),
        component,
    ], style={"flex": "1", "minWidth": "160px", "padding": "0 6px"})


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                title="S&OP Intelligence System")
server = app.server


def header_bar() -> html.Div:
    insights = DATA.get("insights")
    critical = int((insights["Severity"] == "Critical").sum()) if not insights.empty else 0
    return html.Div([
        html.Div([
            html.Div("S&OP INTELLIGENCE SYSTEM",
                     style={"fontSize": "19px", "fontWeight": 700,
                            "color": "white", "letterSpacing": "0.04em"}),
            html.Div("Demand & Supply Planning Command Centre  ·  "
                     "Synthetic data for portfolio demonstration",
                     style={"fontSize": "11px", "color": "#B9C6DE", "marginTop": "2px"}),
        ]),
        html.Div([
            html.Span(f"{critical} CRITICAL ALERTS" if critical else "NO CRITICAL ALERTS",
                      style={"backgroundColor": RED if critical else GREEN,
                             "color": "white", "padding": "6px 14px",
                             "borderRadius": "20px", "fontSize": "11px",
                             "fontWeight": 700, "letterSpacing": "0.05em"}),
        ]),
    ], style={"backgroundColor": NAVY, "padding": "16px 26px",
              "display": "flex", "justifyContent": "space-between",
              "alignItems": "center"})


app.layout = html.Div([
    header_bar(),
    dcc.Tabs(id="tabs", value="tab-forecast", children=[
        dcc.Tab(label="1 · Forecast vs Actual", value="tab-forecast"),
        dcc.Tab(label="2 · Consensus Plan", value="tab-consensus"),
        dcc.Tab(label="3 · Supply Constraints", value="tab-supply"),
        dcc.Tab(label="4 · OTIF & SPC", value="tab-otif"),
        dcc.Tab(label="5 · Executive Insights", value="tab-insights"),
    ], style={"backgroundColor": SURFACE}),
    html.Div(id="tab-content", style={"padding": "18px 26px"}),
], style={"backgroundColor": CANVAS, "minHeight": "100vh",
          "fontFamily": "Inter, Segoe UI, Helvetica, Arial, sans-serif"})


# --------------------------------------------------------------------------- #
# TAB 1 -- Forecast vs Actual
# --------------------------------------------------------------------------- #
def tab_forecast() -> html.Div:
    forecast = DATA.get("forecast")
    hierarchy = DATA.get("hierarchy")
    if forecast.empty:
        return panel("Forecast", html.Div("Run `python run_demo.py` to generate data."))

    categories = sorted(forecast[Keys.CATEGORY].dropna().unique())
    channels = sorted(forecast[Keys.CHANNEL].dropna().unique())

    cards = []
    if not hierarchy.empty:
        review = hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
        sku_week = hierarchy[hierarchy["Aggregation_Level"] == "SKU / Week"]
        if not review.empty:
            wape = float(review.iloc[0]["WAPE"])
            cards.append(kpi_card("Review-Level WAPE", f"{wape:.1f}%",
                                  "Category / Month (S&OP sign-off level)",
                                  "good" if wape < 15 else "warn"))
        if not sku_week.empty:
            row = sku_week.iloc[0]
            cards.append(kpi_card("SKU-Week WAPE", f"{float(row['WAPE']):.1f}%",
                                  "Hardest cut; error does not cancel", "warn"))
            fva = float(row.get("FVA_vs_Naive", 0) or 0)
            cards.append(kpi_card("Forecast Value Added", f"{fva:+.1f} pts",
                                  "vs seasonal-naive benchmark",
                                  "good" if fva > 0 else "bad"))
    accuracy = DATA.get("sku_accuracy")
    if not accuracy.empty and "Selected_Model" in accuracy.columns:
        mix = accuracy["Selected_Model"].value_counts()
        cards.append(kpi_card("Champion Model", str(mix.index[0]),
                              f"{mix.iloc[0] / mix.sum() * 100:.0f}% of portfolio"))
    cards.append(kpi_card("SKUs Forecast", f"{forecast[Keys.SKU].nunique():,}",
                          f"{int(forecast['horizon'].max())}-week horizon"))

    return html.Div([
        html.Div(cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px",
                               "flexWrap": "wrap"}),
        panel("Filters", html.Div([
            control("Category", dropdown("fc-category", categories, placeholder="All categories")),
            control("Channel", dropdown("fc-channel", channels, placeholder="All channels")),
            control("SKU", dropdown("fc-sku", [], placeholder="Select a SKU")),
            control("Horizon (weeks)", dcc.Slider(
                id="fc-horizon", min=1, max=int(forecast["horizon"].max()),
                step=1, value=int(forecast["horizon"].max()),
                marks={i: str(i) for i in range(1, int(forecast["horizon"].max()) + 1, 2)})),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap"})),

        panel("Demand history, forecast and prediction intervals",
              dcc.Graph(id="fc-chart"),
              "Shaded bands are the 80% and 95% prediction intervals, calibrated on "
              "cross-validation residuals and validated on an untouched holdout."),

        html.Div([
            html.Div(panel("Accuracy by aggregation level", dcc.Graph(id="fc-hierarchy"),
                           "Forecast error cancels as demand aggregates. Quoting a single "
                           "number without its level is meaningless."),
                     style={"flex": "1", "minWidth": "380px"}),
            html.Div(panel("Model competition (holdout)", dcc.Graph(id="fc-models"),
                           "Every model scored on the same untouched window."),
                     style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.Div(panel("ML feature importance", dcc.Graph(id="fc-importance"),
                           "Gain-based importance from the global XGBoost model."),
                     style={"flex": "1", "minWidth": "380px"}),
            html.Div(panel("SHAP attribution", dcc.Graph(id="fc-shap"),
                           "Mean absolute SHAP value: average contribution per prediction."),
                     style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        panel("SKU accuracy detail", data_table(
            DATA.get("sku_accuracy").head(300), "fc-table", page_size=10,
            highlight={"Accuracy_Band": {"Poor (>35%)": RED,
                                         "Acceptable (20-35%)": AMBER,
                                         "Good (10-20%)": BLUE,
                                         "Excellent (<10%)": GREEN}})),
    ])


@app.callback(Output("fc-sku", "options"),
              Input("fc-category", "value"), Input("fc-channel", "value"))
def _fc_sku_options(category, channel):
    forecast = DATA.get("forecast")
    if forecast.empty:
        return []
    subset = forecast
    if category:
        subset = subset[subset[Keys.CATEGORY] == category]
    if channel:
        subset = subset[subset[Keys.CHANNEL] == channel]
    skus = sorted(subset[Keys.SKU].unique())[:400]
    return [{"label": s, "value": s} for s in skus]


@app.callback(Output("fc-chart", "figure"),
              Input("fc-sku", "value"), Input("fc-category", "value"),
              Input("fc-channel", "value"), Input("fc-horizon", "value"))
def _fc_chart(sku, category, channel, horizon):
    forecast = DATA.get("forecast")
    if forecast.empty:
        return empty_figure("No forecast data available.")

    subset = forecast.copy()
    if category:
        subset = subset[subset[Keys.CATEGORY] == category]
    if channel:
        subset = subset[subset[Keys.CHANNEL] == channel]
    if horizon:
        subset = subset[subset["horizon"] <= horizon]
    if subset.empty:
        return empty_figure("No SKUs match the selected filters.")

    fig = go.Figure()
    if sku and sku in set(subset[Keys.SKU]):
        series = subset[subset[Keys.SKU] == sku].sort_values("horizon")
        title = f"{sku} — {series[Keys.CATEGORY].iloc[0]} / {series[Keys.CHANNEL].iloc[0]}"
        x = series["horizon"]
        fig.add_trace(go.Scatter(x=x, y=series["P97_5"], line=dict(width=0),
                                 name="95% interval", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=series["P2_5"], fill="tonexty",
                                 fillcolor="rgba(46,117,182,0.10)", line=dict(width=0),
                                 name="95% interval"))
        fig.add_trace(go.Scatter(x=x, y=series["P90"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=series["P10"], fill="tonexty",
                                 fillcolor="rgba(46,117,182,0.22)", line=dict(width=0),
                                 name="80% interval"))
        fig.add_trace(go.Scatter(x=x, y=series["Forecast_Units"], mode="lines+markers",
                                 line=dict(color=NAVY, width=3), name="Forecast"))
        promo = series[series.get("Is_Promo", 0) == 1]
        if not promo.empty:
            fig.add_trace(go.Scatter(x=promo["horizon"], y=promo["Forecast_Units"],
                                     mode="markers", name="Promotion planned",
                                     marker=dict(color=AMBER, size=12, symbol="star")))
    else:
        agg = (subset.groupby("horizon", as_index=False)
               .agg(Forecast_Units=("Forecast_Units", "sum"),
                    P10=("P10", "sum"), P90=("P90", "sum")))
        label = category or channel or "Total portfolio"
        title = f"{label} — aggregated forecast ({subset[Keys.SKU].nunique()} SKUs)"
        fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P90"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P10"], fill="tonexty",
                                 fillcolor="rgba(46,117,182,0.20)", line=dict(width=0),
                                 name="80% interval"))
        fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Forecast_Units"],
                             marker_color=BLUE, name="Forecast", opacity=0.85))
    fig.update_layout(**PLOT_LAYOUT, title=title,
                      xaxis_title="Forecast week", yaxis_title="Units")
    return fig


@app.callback(Output("fc-hierarchy", "figure"), Input("tabs", "value"))
def _fc_hierarchy(_):
    hierarchy = DATA.get("hierarchy")
    if hierarchy.empty:
        return empty_figure("No accuracy data available.")
    frame = hierarchy.sort_values("WAPE")
    colours = [GREEN if v < 15 else AMBER if v < 30 else RED for v in frame["WAPE"]]
    fig = go.Figure(go.Bar(x=frame["WAPE"], y=frame["Aggregation_Level"],
                           orientation="h", marker_color=colours,
                           text=[f"{v:.1f}%" for v in frame["WAPE"]],
                           textposition="outside"))
    fig.add_vline(x=CONFIG.forecast.portfolio_mape_target, line_dash="dash",
                  line_color=NAVY,
                  annotation_text=f"{CONFIG.forecast.portfolio_mape_target:.0f}% benchmark")
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="WAPE (%)", yaxis_title="")
    return fig


@app.callback(Output("fc-models", "figure"), Input("tabs", "value"))
def _fc_models(_):
    leaderboard = DATA.get("leaderboard")
    if leaderboard.empty:
        return empty_figure("No model comparison available.")
    holdout = leaderboard[leaderboard["Stage"] == "HOLDOUT"]
    if holdout.empty:
        holdout = leaderboard
    fig = px.bar(holdout, x="Model", y="Median_MAPE", color="Tier",
                 barmode="group",
                 color_discrete_map={"Competition Panel": NAVY, "Full Portfolio": BLUE})
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="", yaxis_title="Median MAPE (%)")
    return fig


@app.callback(Output("fc-importance", "figure"), Input("tabs", "value"))
def _fc_importance(_):
    importance = DATA.get("importance")
    if importance.empty:
        return empty_figure("Feature importance unavailable (XGBoost not fitted).")
    top = importance.head(15).sort_values("Importance")
    fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"],
                           orientation="h", marker_color=BLUE))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Gain importance", yaxis_title="")
    return fig


@app.callback(Output("fc-shap", "figure"), Input("tabs", "value"))
def _fc_shap(_):
    shap_df = DATA.get("shap")
    if shap_df.empty:
        return empty_figure("SHAP attribution unavailable in this environment.")
    top = shap_df.head(15).sort_values("Mean_Abs_SHAP")
    fig = go.Figure(go.Bar(x=top["Mean_Abs_SHAP"], y=top["Feature"],
                           orientation="h", marker_color=NAVY))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Mean |SHAP| (units)", yaxis_title="")
    return fig


# --------------------------------------------------------------------------- #
# TAB 2 -- Consensus Plan
# --------------------------------------------------------------------------- #
def tab_consensus() -> html.Div:
    consensus = DATA.get("consensus")
    scenarios = DATA.get("scenarios")
    if consensus.empty:
        return panel("Consensus", html.Div("Run `python run_demo.py` first."))

    available = sorted(consensus["Scenario"].dropna().unique())
    default = (CONFIG.consensus.default_scenario if CONFIG.consensus.default_scenario
               in available else available[0])
    categories = sorted(consensus[Keys.CATEGORY].dropna().unique())

    return html.Div([
        html.Div(id="cons-kpis", style={"display": "flex", "gap": "12px",
                                        "marginBottom": "16px", "flexWrap": "wrap"}),
        panel("Scenario and filters", html.Div([
            control("Planning scenario", dropdown("cons-scenario", available, default)),
            control("Category", dropdown("cons-category", categories,
                                         placeholder="All categories")),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap"})),

        panel("Consensus bridge: statistical baseline to final plan",
              dcc.Graph(id="cons-waterfall"),
              "Each step is a delta. The statistical baseline is never overwritten — it is "
              "carried through so plan misses can be attributed to model or to override."),

        html.Div([
            html.Div(panel("Scenario comparison", dcc.Graph(id="cons-scenarios"),
                           "Scenario choice is a working-capital and service decision."),
                     style={"flex": "1", "minWidth": "380px"}),
            html.Div(panel("Override distribution", dcc.Graph(id="cons-overrides"),
                           f"Governance threshold: "
                           f"{CONFIG.consensus.override_alert_threshold_pct:.0f}% deviation."),
                     style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        panel("Weekly plan: statistical, commercial and constrained",
              dcc.Graph(id="cons-weekly")),

        panel("Override audit log — flagged items requiring justification",
              html.Div(id="cons-table"),
              "Every override carries its driver, justification, submitter and a "
              "governance status. Flagged items without a valid justification are blocked."),
    ])


@app.callback(Output("cons-kpis", "children"),
              Input("cons-scenario", "value"))
def _cons_kpis(scenario):
    consensus = DATA.get("consensus")
    if consensus.empty or not scenario:
        return []
    subset = consensus[consensus["Scenario"] == scenario]
    statistical = subset["Forecast_Units"].sum()
    final = subset["Consensus_Units"].sum()
    uplift = (final - statistical) / statistical * 100 if statistical else 0
    alerts = int(subset["Override_Alert"].sum())
    constrained = float(subset["Constrained_Out_Value"].sum())
    return [
        kpi_card("Consensus Volume", f"{final / 1e6:.2f}M", "units over the horizon"),
        kpi_card("Consensus Value",
                 human_number(subset["Consensus_Value"].sum(), RC.currency_symbol),
                 "at list price"),
        kpi_card("Uplift vs Statistical", f"{uplift:+.1f}%",
                 "commercial + constraint effect",
                 "warn" if abs(uplift) > 12 else "good"),
        kpi_card("Override Alerts", f"{alerts:,}",
                 f">{CONFIG.consensus.override_alert_threshold_pct:.0f}% deviation",
                 "warn" if alerts else "good"),
        kpi_card("Constrained Out", human_number(constrained, RC.currency_symbol),
                 "demand capacity cannot serve", "bad" if constrained > 0 else "good"),
    ]


@app.callback(Output("cons-waterfall", "figure"),
              Input("cons-scenario", "value"), Input("cons-category", "value"))
def _cons_waterfall(scenario, category):
    consensus = DATA.get("consensus")
    if consensus.empty or not scenario:
        return empty_figure("No consensus plan available.")
    subset = consensus[consensus["Scenario"] == scenario]
    if category:
        subset = subset[subset[Keys.CATEGORY] == category]
    if subset.empty:
        return empty_figure("No rows match the filters.")

    statistical = subset["Forecast_Units"].sum()
    promo_drivers = {"Promotion Uplift", "New Product Introduction"}
    delta_all = subset["Commercial_Forecast"] - subset["Forecast_Units"]
    is_promo = subset["Commercial_Driver"].isin(promo_drivers)
    accepted = subset["Unconstrained_Plan"].sum() - statistical
    raw = delta_all.sum()
    scale = accepted / raw if abs(raw) > 1e-9 else 0.0
    constraint = -(subset["Unconstrained_Plan"].sum() - subset["Consensus_Units"].sum())

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Statistical<br>Baseline", "Commercial<br>Override", "Promotion<br>/ NPD",
           "Operational<br>Constraint", "Final<br>Consensus"],
        y=[statistical, delta_all[~is_promo].sum() * scale,
           delta_all[is_promo].sum() * scale, constraint, 0],
        connector={"line": {"color": MUTED}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": NAVY}},
        text=[f"{v:,.0f}" for v in [statistical, delta_all[~is_promo].sum() * scale,
                                    delta_all[is_promo].sum() * scale, constraint,
                                    subset["Consensus_Units"].sum()]],
        textposition="outside"))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "x"},
                      title=f"{scenario} scenario"
                            + (f" — {category}" if category else ""),
                      yaxis_title="Units")
    return fig


@app.callback(Output("cons-scenarios", "figure"), Input("cons-scenario", "value"))
def _cons_scenarios(_):
    scenarios = DATA.get("scenarios")
    if scenarios.empty:
        return empty_figure("No scenario comparison available.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenarios["Scenario"], y=scenarios["Consensus_Units"],
                         name="Consensus units", marker_color=NAVY))
    fig.add_trace(go.Bar(x=scenarios["Scenario"], y=scenarios["Constrained_Out_Units"],
                         name="Constrained out", marker_color=RED))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "x"}, barmode="stack",
                      xaxis_title="", yaxis_title="Units")
    return fig


@app.callback(Output("cons-overrides", "figure"), Input("cons-scenario", "value"))
def _cons_overrides(scenario):
    consensus = DATA.get("consensus")
    if consensus.empty or not scenario:
        return empty_figure("No override data available.")
    subset = consensus[(consensus["Scenario"] == scenario)
                       & consensus["Has_Commercial_Input"]]
    if subset.empty:
        return empty_figure("No commercial overrides submitted.")
    threshold = CONFIG.consensus.override_alert_threshold_pct
    fig = go.Figure(go.Histogram(x=subset["Override_Deviation_Pct"], nbinsx=50,
                                 marker_color=BLUE))
    for sign in (1, -1):
        fig.add_vline(x=sign * threshold, line_dash="dash", line_color=RED)
    fig.add_annotation(x=threshold, y=1, yref="paper",
                       text=f"±{threshold:.0f}% governance threshold",
                       showarrow=False, font=dict(size=10, color=RED))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Deviation from statistical baseline (%)",
                      yaxis_title="SKU-weeks")
    return fig


@app.callback(Output("cons-weekly", "figure"),
              Input("cons-scenario", "value"), Input("cons-category", "value"))
def _cons_weekly(scenario, category):
    consensus = DATA.get("consensus")
    if consensus.empty or not scenario:
        return empty_figure("No consensus plan available.")
    subset = consensus[consensus["Scenario"] == scenario]
    if category:
        subset = subset[subset[Keys.CATEGORY] == category]
    if subset.empty:
        return empty_figure("No rows match the filters.")
    agg = (subset.groupby("horizon", as_index=False)
           .agg(Statistical=("Forecast_Units", "sum"),
                Commercial=("Commercial_Forecast", "sum"),
                Consensus=("Consensus_Units", "sum"),
                Constrained_Out=("Constrained_Out_Units", "sum")))
    fig = go.Figure()
    for name, colour, dash_style in [("Statistical", MUTED, "dot"),
                                     ("Commercial", AMBER, "dash"),
                                     ("Consensus", NAVY, None)]:
        fig.add_trace(go.Scatter(x=agg["horizon"], y=agg[name], name=name,
                                 mode="lines+markers",
                                 line=dict(color=colour, width=3, dash=dash_style)))
    fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Constrained_Out"],
                         name="Constrained out", marker_color=RED, opacity=0.45))
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Forecast week", yaxis_title="Units")
    return fig


@app.callback(Output("cons-table", "children"), Input("cons-scenario", "value"))
def _cons_table(scenario):
    log = DATA.get("override_log")
    if log.empty:
        return html.Div("No overrides recorded.", style={"color": MUTED})
    subset = log[log["Scenario"] == scenario] if "Scenario" in log.columns else log
    flagged = subset[subset["Override_Alert"]] if "Override_Alert" in subset.columns else subset
    display = flagged if not flagged.empty else subset
    keep = [c for c in [Keys.SKU, Keys.CATEGORY, "horizon", "Statistical_Forecast",
                        "Commercial_Override", "Override_Deviation_Pct",
                        "Commercial_Driver", "Submitted_By", "Governance_Status",
                        "Planner_Decision"] if c in display.columns]
    return data_table(display[keep].head(300), "cons-log-table", page_size=10,
                      highlight={"Governance_Status": {
                          "BLOCKED - Justification Missing": RED,
                          "Flagged - Justified": AMBER,
                          "Within Tolerance": GREEN}})


# --------------------------------------------------------------------------- #
# TAB 3 -- Supply Constraints
# --------------------------------------------------------------------------- #
def tab_supply() -> html.Div:
    summary = DATA.get("supply_summary")
    plan = DATA.get("supply_plan")
    capacity = DATA.get("capacity")
    if summary.empty:
        return panel("Supply", html.Div("Run `python run_demo.py` first."))

    bands = summary["Risk_Band"].value_counts()
    breaches = int((capacity["Severity"] == "BREACH").sum()) if not capacity.empty else 0
    expedite = int((summary["First_Stockout_Week"].notna()
                    & ~summary["Can_Recover_In_Lead_Time"].astype(bool)).sum())
    categories = sorted(summary[Keys.CATEGORY].dropna().unique())

    cards = [
        kpi_card("Red Risk SKUs", f"{int(bands.get('Red', 0)):,}",
                 f"of {len(summary):,} planned",
                 "bad" if bands.get("Red", 0) > 50 else "warn"),
        kpi_card("Amber Risk SKUs", f"{int(bands.get('Amber', 0)):,}", "monitor", "warn"),
        kpi_card("Capacity Breaches", f"{breaches:,}", "resource-weeks over 100%",
                 "bad" if breaches else "good"),
        kpi_card("Expedite Required", f"{expedite:,}", "stockout inside lead time",
                 "bad" if expedite else "good"),
        kpi_card("Avg Weeks of Cover",
                 f"{plan['Weeks_Of_Cover'].mean():.1f}" if not plan.empty else "n/a",
                 f"target {CONFIG.supply.target_weeks_of_cover:.0f} weeks"),
    ]

    return html.Div([
        html.Div(cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px",
                               "flexWrap": "wrap"}),
        panel("Filters", html.Div([
            control("Category", dropdown("sup-category", categories,
                                         placeholder="All categories")),
            control("Risk band", dropdown("sup-band", ["Red", "Amber", "Green"],
                                          placeholder="All bands")),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap"})),

        html.Div([
            html.Div(panel("Supplier capacity utilisation heatmap",
                           dcc.Graph(id="sup-heatmap"),
                           "Rows are suppliers, columns are horizon weeks. Anything above "
                           "100% is a commitment the supply base cannot physically meet."),
                     style={"flex": "1.4", "minWidth": "420px"}),
            html.Div(panel("Supply Risk Index distribution", dcc.Graph(id="sup-risk"),
                           "Weighted blend of five documented, inspectable factors."),
                     style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        panel("Projected inventory trajectory", dcc.Graph(id="sup-trajectory"),
              "Aggregate closing inventory against the aggregate safety-stock line."),

        html.Div([
            html.Div(panel("Weeks of cover vs risk", dcc.Graph(id="sup-scatter"),
                           "Bubble size is 12-week demand. The bottom-right quadrant is "
                           "where a planner should spend the morning."),
                     style={"flex": "1", "minWidth": "380px"}),
            html.Div(panel("Stockout timing", dcc.Graph(id="sup-stockouts"),
                           "How much notice the plan gives before each shortage."),
                     style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        panel("Capacity constraint alert log", html.Div(id="sup-capacity-table")),
        panel("SKU risk register", html.Div(id="sup-table")),
    ])


def _filter_supply(category, band) -> pd.DataFrame:
    summary = DATA.get("supply_summary")
    if summary.empty:
        return summary
    subset = summary
    if category:
        subset = subset[subset[Keys.CATEGORY] == category]
    if band:
        subset = subset[subset["Risk_Band"] == band]
    return subset


@app.callback(Output("sup-heatmap", "figure"),
              Input("sup-category", "value"), Input("sup-band", "value"))
def _sup_heatmap(category, band):
    capacity = DATA.get("capacity")
    if capacity.empty:
        return empty_figure("No capacity data available.")
    supplier = capacity[capacity["Resource_Type"] == "Supplier"]
    if supplier.empty:
        return empty_figure("No supplier capacity records.")
    pivot = supplier.pivot_table(index="Resource_ID", columns="horizon",
                                 values="Utilisation_Pct", aggfunc="max")
    # Show the suppliers that actually matter: highest peak utilisation first.
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index[:25]]
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=[f"W{c}" for c in pivot.columns], y=pivot.index,
        colorscale=[[0, "#E8F3EC"], [0.5, "#FFF3CD"], [0.75, "#F8C9A0"], [1, RED]],
        zmin=0, zmax=150, colorbar=dict(title="Util %"),
        hovertemplate="%{y} · %{x}<br>Utilisation: %{z:.0f}%<extra></extra>"))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Horizon week", yaxis_title="")
    return fig


@app.callback(Output("sup-risk", "figure"),
              Input("sup-category", "value"), Input("sup-band", "value"))
def _sup_risk(category, band):
    subset = _filter_supply(category, band)
    if subset.empty:
        return empty_figure("No SKUs match the filters.")
    counts = subset["Risk_Band"].value_counts().reindex(["Green", "Amber", "Red"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(),
                           marker_color=[GREEN, AMBER, RED],
                           text=[f"{int(v)}" for v in counts.to_numpy()],
                           textposition="outside"))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="", yaxis_title="SKUs")
    return fig


@app.callback(Output("sup-trajectory", "figure"), Input("sup-category", "value"))
def _sup_trajectory(category):
    plan = DATA.get("supply_plan")
    if plan.empty:
        return empty_figure("No supply plan available.")
    subset = plan[plan[Keys.CATEGORY] == category] if category else plan
    if subset.empty:
        return empty_figure("No rows match the filters.")
    agg = (subset.groupby("horizon", as_index=False)
           .agg(Closing=("Closing_Inventory", "sum"),
                Safety=("Safety_Stock", "sum"),
                Demand=("Demand_Units", "sum"),
                Receipts=("Planned_Receipts", "sum")))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Demand"], name="Demand",
                         marker_color=LIGHT))
    fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Receipts"], name="Planned receipts",
                         marker_color=BLUE, opacity=0.7))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["Closing"], name="Closing inventory",
                             mode="lines+markers", line=dict(color=NAVY, width=3)))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["Safety"], name="Safety stock",
                             mode="lines", line=dict(color=RED, width=2, dash="dash")))
    fig.update_layout(**PLOT_LAYOUT, barmode="group",
                      xaxis_title="Horizon week", yaxis_title="Units")
    return fig


@app.callback(Output("sup-scatter", "figure"),
              Input("sup-category", "value"), Input("sup-band", "value"))
def _sup_scatter(category, band):
    subset = _filter_supply(category, band)
    if subset.empty:
        return empty_figure("No SKUs match the filters.")
    plot = subset.copy()
    plot["Min_Cover"] = plot["Min_Cover"].clip(upper=20)
    plot["Size"] = plot["Total_Demand"].clip(lower=1)
    fig = px.scatter(plot, x="Min_Cover", y="Supply_Risk_Index", color="Risk_Band",
                     size="Size", size_max=30, hover_name=Keys.SKU,
                     hover_data={Keys.CATEGORY: True, Keys.SUPPLIER: True, "Size": False},
                     color_discrete_map=RAG)
    fig.add_hline(y=CONFIG.supply.risk_red_threshold, line_dash="dash", line_color=RED)
    fig.add_vline(x=CONFIG.supply.target_weeks_of_cover, line_dash="dash", line_color=NAVY)
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Minimum weeks of cover",
                      yaxis_title="Supply Risk Index")
    return fig


@app.callback(Output("sup-stockouts", "figure"), Input("sup-category", "value"))
def _sup_stockouts(category):
    stockouts = DATA.get("stockouts")
    if stockouts.empty:
        return empty_figure("No projected stockouts — the plan is feasible.")
    subset = stockouts[stockouts[Keys.CATEGORY] == category] if category else stockouts
    projected = subset[subset["Event_Type"] == "Projected Stockout"]
    if projected.empty:
        return empty_figure("No hard stockouts projected in the horizon.")
    counts = projected.groupby("horizon").size()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(), marker_color=RED))
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="Week of projected stockout",
                      yaxis_title="SKU-week events")
    return fig


@app.callback(Output("sup-capacity-table", "children"), Input("sup-category", "value"))
def _sup_capacity_table(_):
    capacity = DATA.get("capacity")
    if capacity.empty:
        return html.Div("No capacity data.", style={"color": MUTED})
    flagged = capacity[capacity["Severity"] != "OK"]
    display = flagged if not flagged.empty else capacity.head(25)
    keep = [c for c in ["Resource_Type", "Resource_ID", "horizon", "Required_Units",
                        "Available_Capacity", "Utilisation_Pct", "Overload_Units",
                        "SKUs_Affected", "Severity", "Recommended_Action"]
            if c in display.columns]
    return data_table(display[keep].head(300), "sup-cap-table", page_size=8,
                      highlight={"Severity": {"BREACH": RED, "WARNING": AMBER, "OK": GREEN}})


@app.callback(Output("sup-table", "children"),
              Input("sup-category", "value"), Input("sup-band", "value"))
def _sup_table(category, band):
    subset = _filter_supply(category, band)
    if subset.empty:
        return html.Div("No SKUs match the filters.", style={"color": MUTED})
    keep = [c for c in [Keys.SKU, Keys.CATEGORY, Keys.SUPPLIER, "ABC_XYZ",
                        "Supply_Risk_Index", "Risk_Band", "Min_Cover",
                        "First_Stockout_Week", "Lead_Time_Weeks",
                        "Can_Recover_In_Lead_Time", "Recommended_Action"]
            if c in subset.columns]
    return data_table(subset[keep].head(300), "sup-risk-table", page_size=10,
                      highlight={"Risk_Band": RAG})


# --------------------------------------------------------------------------- #
# TAB 4 -- OTIF & SPC
# --------------------------------------------------------------------------- #
def tab_otif() -> html.Div:
    month = DATA.get("otif_month")
    spc = DATA.get("spc")
    if month.empty:
        return panel("OTIF", html.Div("Run `python run_demo.py` first."))

    supplier = DATA.get("otif_supplier")
    overall = (month["OTIF_Orders"].sum() / month["Orders"].sum() * 100
               if month["Orders"].sum() else 0)
    status = ("good" if overall >= CONFIG.kpi.otif_world_class
              else "warn" if overall >= CONFIG.kpi.otif_acceptable else "bad")
    special = int(spc["Special_Cause"].sum()) if not spc.empty else 0

    cards = [
        kpi_card("Overall OTIF", f"{overall:.1f}%",
                 f"benchmark {CONFIG.kpi.otif_acceptable:.0f}% acceptable", status),
        kpi_card("On Time",
                 f"{month['On_Time_Orders'].sum() / month['Orders'].sum() * 100:.1f}%",
                 "delivered by promised date"),
        kpi_card("In Full",
                 f"{month['In_Full_Orders'].sum() / month['Orders'].sum() * 100:.1f}%",
                 f"≥{CONFIG.kpi.in_full_tolerance * 100:.0f}% of ordered quantity"),
        kpi_card("Special-Cause Months", f"{special}",
                 "SPC signal, not routine variation",
                 "bad" if special else "good"),
        kpi_card("Suppliers At Risk",
                 f"{int((supplier['Benchmark'] == 'At Risk').sum()):,}"
                 if not supplier.empty else "n/a",
                 f"below {CONFIG.kpi.otif_acceptable:.0f}% OTIF", "warn"),
    ]

    return html.Div([
        html.Div(cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px",
                               "flexWrap": "wrap"}),

        panel("Six Sigma individuals control chart — monthly OTIF",
              dcc.Graph(id="otif-spc"),
              "Sigma is estimated from the average moving range (MRbar / 1.128), not the raw "
              "standard deviation, so the limits are not inflated by the special causes they "
              "exist to detect. Points outside the limits are process changes; points inside "
              "are noise and must not be reacted to."),

        panel("Western Electric rule violations", html.Div(id="otif-violations"),
              "Beyond the 3-sigma test, these rules detect sustained shifts and trends that "
              "sit inside the limits but are collectively improbable."),

        html.Div([
            html.Div(panel("OTIF by category", dcc.Graph(id="otif-category")),
                     style={"flex": "1", "minWidth": "320px"}),
            html.Div(panel("OTIF by channel", dcc.Graph(id="otif-channel")),
                     style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.Div(panel("Root-cause Pareto", dcc.Graph(id="otif-pareto"),
                           "Ranked by value at risk, not frequency — the most common "
                           "failure is rarely the most expensive."),
                     style={"flex": "1.3", "minWidth": "400px"}),
            html.Div(panel("Worst-performing suppliers", dcc.Graph(id="otif-suppliers")),
                     style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        panel("Supplier scorecard", data_table(
            DATA.get("scorecard").head(200), "otif-table", page_size=10,
            highlight={"Benchmark": RAG,
                       "Trend": {"Deteriorating": RED, "Improving": GREEN,
                                 "Stable": BLUE}})),
    ])


@app.callback(Output("otif-spc", "figure"), Input("tabs", "value"))
def _otif_spc(_):
    spc = DATA.get("spc")
    if spc.empty:
        return empty_figure("No SPC chart available.")
    x = spc["Order_Month"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=spc["UCL"], name="UCL (+3σ)", mode="lines",
                             line=dict(color=RED, dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=spc["LCL"], name="LCL (−3σ)", mode="lines",
                             line=dict(color=RED, dash="dash", width=1.5),
                             fill="tonexty", fillcolor="rgba(15,123,79,0.05)"))
    fig.add_trace(go.Scatter(x=x, y=spc["Center_Line"], name="Centre line",
                             mode="lines", line=dict(color=NAVY, width=2)))
    if "UCL_2Sigma" in spc.columns:
        for col, label in [("UCL_2Sigma", "+2σ"), ("LCL_2Sigma", "−2σ")]:
            fig.add_trace(go.Scatter(x=x, y=spc[col], name=label, mode="lines",
                                     line=dict(color=AMBER, dash="dot", width=1),
                                     showlegend=False))

    normal = spc[~spc["Special_Cause"].astype(bool)]
    special = spc[spc["Special_Cause"].astype(bool)]
    fig.add_trace(go.Scatter(x=normal["Order_Month"], y=normal["OTIF_Pct"],
                             name="In control", mode="lines+markers",
                             line=dict(color=BLUE, width=2.5),
                             marker=dict(size=9, color=BLUE)))
    if not special.empty:
        fig.add_trace(go.Scatter(x=special["Order_Month"], y=special["OTIF_Pct"],
                                 name="Special cause", mode="markers",
                                 marker=dict(size=16, color=RED, symbol="x",
                                             line=dict(width=2))))
    fig.add_hline(y=CONFIG.kpi.otif_acceptable, line_dash="dot", line_color=MUTED,
                  annotation_text=f"{CONFIG.kpi.otif_acceptable:.0f}% business threshold")
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Month", yaxis_title="OTIF (%)")
    return fig


@app.callback(Output("otif-violations", "children"), Input("tabs", "value"))
def _otif_violations(_):
    violations = DATA.get("spc_violations")
    if violations.empty:
        return html.Div("No Western Electric rule violations detected — the process is "
                        "statistically stable.", style={"color": GREEN, "fontWeight": 600})
    return data_table(violations, "otif-viol-table", page_size=8)


@app.callback(Output("otif-category", "figure"), Input("tabs", "value"))
def _otif_category(_):
    frame = DATA.get("otif_category")
    if frame.empty:
        return empty_figure("No category OTIF available.")
    frame = frame.sort_values("OTIF_Pct")
    colours = [RAG.get(b, BLUE) for b in frame["Benchmark"]]
    fig = go.Figure(go.Bar(x=frame["OTIF_Pct"], y=frame[Keys.CATEGORY],
                           orientation="h", marker_color=colours,
                           text=[f"{v:.1f}%" for v in frame["OTIF_Pct"]],
                           textposition="outside"))
    fig.add_vline(x=CONFIG.kpi.otif_acceptable, line_dash="dash", line_color=NAVY)
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="OTIF (%)", yaxis_title="")
    return fig


@app.callback(Output("otif-channel", "figure"), Input("tabs", "value"))
def _otif_channel(_):
    frame = DATA.get("otif_channel")
    if frame.empty:
        return empty_figure("No channel OTIF available.")
    frame = frame.sort_values("OTIF_Pct")
    colours = [RAG.get(b, BLUE) for b in frame["Benchmark"]]
    fig = go.Figure(go.Bar(x=frame["OTIF_Pct"], y=frame[Keys.CHANNEL],
                           orientation="h", marker_color=colours,
                           text=[f"{v:.1f}%" for v in frame["OTIF_Pct"]],
                           textposition="outside"))
    fig.add_vline(x=CONFIG.kpi.otif_acceptable, line_dash="dash", line_color=NAVY)
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="OTIF (%)", yaxis_title="")
    return fig


@app.callback(Output("otif-pareto", "figure"), Input("tabs", "value"))
def _otif_pareto(_):
    causes = DATA.get("root_cause")
    if causes.empty:
        return empty_figure("No service failures recorded.")
    frame = causes.sort_values("Value_At_Risk", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame["Root_Cause"], y=frame["Value_At_Risk"],
                         name="Value at risk",
                         marker_color=[RED if v else BLUE
                                       for v in frame.get("Vital_Few", [False] * len(frame))]))
    if "Cumulative_Value_Pct" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["Root_Cause"], y=frame["Cumulative_Value_Pct"],
                                 name="Cumulative %", yaxis="y2", mode="lines+markers",
                                 line=dict(color=NAVY, width=3)))
        fig.add_hline(y=80, line_dash="dash", line_color=MUTED, yref="y2")
    layout = {**PLOT_LAYOUT, "hovermode": "x"}
    fig.update_layout(**layout, yaxis_title=f"Value at risk ({RC.currency_symbol})",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                  title="Cumulative %"),
                      xaxis_title="")
    return fig


@app.callback(Output("otif-suppliers", "figure"), Input("tabs", "value"))
def _otif_suppliers(_):
    frame = DATA.get("otif_supplier")
    if frame.empty:
        return empty_figure("No supplier OTIF available.")
    reliable = frame[frame["Statistically_Reliable"]] if "Statistically_Reliable" in frame else frame
    worst = reliable.nsmallest(12, "OTIF_Pct").sort_values("OTIF_Pct")
    colours = [RAG.get(b, BLUE) for b in worst["Benchmark"]]
    fig = go.Figure(go.Bar(x=worst["OTIF_Pct"], y=worst[Keys.SUPPLIER],
                           orientation="h", marker_color=colours,
                           customdata=worst[["Orders"]],
                           hovertemplate="%{y}<br>OTIF %{x:.1f}%<br>"
                                         "Orders: %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=CONFIG.kpi.otif_acceptable, line_dash="dash", line_color=NAVY)
    fig.update_layout(**{**PLOT_LAYOUT, "hovermode": "closest"},
                      xaxis_title="OTIF (%)", yaxis_title="")
    return fig


# --------------------------------------------------------------------------- #
# TAB 5 -- Executive Insights
# --------------------------------------------------------------------------- #
def tab_insights() -> html.Div:
    insights = DATA.get("insights")
    if insights.empty:
        return panel("Insights", html.Div("Run `python run_demo.py` first."))

    counts = insights["Severity"].value_counts()
    cards = [kpi_card(f"{sev} Findings", f"{int(counts.get(sev, 0))}", "",
                      {"Critical": "bad", "High": "warn", "Medium": "warn"}.get(sev, "good"))
             for sev in ["Critical", "High", "Medium", "Info"]]

    blocks = []
    for severity in ["Critical", "High", "Medium", "Info"]:
        subset = insights[insights["Severity"] == severity]
        if subset.empty:
            continue
        items = []
        for _, row in subset.iterrows():
            items.append(html.Div([
                html.Div([
                    html.Span(row["Domain"], style={
                        "backgroundColor": RAG.get(severity, BLUE), "color": "white",
                        "padding": "2px 9px", "borderRadius": "10px",
                        "fontSize": "10px", "fontWeight": 700,
                        "letterSpacing": "0.04em"}),
                    html.Span(f"  Owner: {row.get('Owner', '')}",
                              style={"fontSize": "11px", "color": MUTED,
                                     "marginLeft": "8px"}),
                ]),
                html.Div(row["Headline"], style={"fontSize": "13px", "fontWeight": 600,
                                                 "color": INK, "margin": "6px 0 4px"}),
                html.Div(row["Detail"], style={"fontSize": "12px", "color": MUTED,
                                               "lineHeight": "1.5"}),
                html.Div(f"→ {row.get('Recommended_Action', '')}",
                         style={"fontSize": "12px", "color": NAVY, "marginTop": "6px",
                                "fontWeight": 600})
                if row.get("Recommended_Action") else None,
            ], style={"padding": "12px 14px", "borderLeft": f"3px solid "
                                                            f"{RAG.get(severity, BLUE)}",
                      "backgroundColor": "#FBFCFE", "marginBottom": "10px",
                      "borderRadius": "0 8px 8px 0"}))
        blocks.append(panel(f"{severity} findings ({len(subset)})", html.Div(items)))

    return html.Div([
        html.Div(cards, style={"display": "flex", "gap": "12px", "marginBottom": "16px",
                               "flexWrap": "wrap"}),
        html.Div("Every finding below is generated from a calculated metric crossing a "
                 "configured threshold. None are hard-coded, and a clean result produces "
                 "no finding rather than a reassuring one.",
                 style={"fontSize": "12px", "color": MUTED, "marginBottom": "14px",
                        "fontStyle": "italic"}),
        *blocks,
        panel("Full insight register", data_table(insights, "ins-table", page_size=15,
                                                  highlight={"Severity": RAG})),
    ])


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def _render_tab(tab: str):
    if not DATA.available():
        return panel("Data not found", html.Div([
            html.P("No processed datasets were found in data/processed."),
            html.P("Run the pipeline first:"),
            html.Pre("python run_demo.py",
                     style={"backgroundColor": "#F1F5F9", "padding": "10px",
                            "borderRadius": "6px"}),
        ]))
    return {
        "tab-forecast": tab_forecast,
        "tab-consensus": tab_consensus,
        "tab-supply": tab_supply,
        "tab-otif": tab_otif,
        "tab-insights": tab_insights,
    }.get(tab, tab_forecast)()


def run_dashboard(host: Optional[str] = None, port: Optional[int] = None,
                  debug: bool = False) -> None:
    host = host or RC.dashboard_host
    port = port or RC.dashboard_port
    logger.info("Dashboard starting at http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()
