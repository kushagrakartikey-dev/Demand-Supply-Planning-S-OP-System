"""
Standalone HTML dashboard export.

Produces a **single self-contained `.html` file** containing the full S&OP
command centre: charts, KPI cards, filterable tables and the insight register.
It opens in any browser with no Python, no server and no install — and it can be
emailed, committed, or published to GitHub Pages.

Why this exists alongside the Dash app
--------------------------------------
The Dash application is the working tool: live callbacks, cross-filtering,
server-side data access. But a Dash app requires someone to run it, which makes
it useless as a *deliverable*. A hiring manager opening a repository will not
`pip install` and launch a server; a supply chain director will not either.

The static export solves the distribution problem without pretending to be the
same artefact. Plotly figures are embedded as interactive JavaScript — zoom,
hover, legend toggling and tab navigation all work offline — while the
server-dependent behaviour (SKU-level drill-down across 500 series) is
deliberately pre-aggregated rather than faked.

Both read the same published CSVs, so the numbers cannot diverge.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.config import OUTPUT_DIR, PROCESSED_DIR, Keys, load_config
from src.shared.utils import get_logger, human_number

logger = get_logger("static_export")

CONFIG = load_config()
RC = CONFIG.report

NAVY = f"#{RC.brand_navy}"
BLUE = f"#{RC.brand_blue}"
LIGHT = f"#{RC.brand_light}"
GREEN = f"#{RC.brand_green}"
AMBER = f"#{RC.brand_amber}"
RED = f"#{RC.brand_red}"
INK = "#1B1F2A"
MUTED = "#6B7280"

RAG = {"Green": GREEN, "Amber": AMBER, "Red": RED, "World Class": GREEN,
       "Acceptable": AMBER, "At Risk": RED, "OK": GREEN, "WARNING": AMBER,
       "BREACH": RED, "Critical": RED, "High": "#C2410C", "Medium": AMBER,
       "Info": BLUE}

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
              size=12, color=INK),
    margin=dict(l=60, r=30, t=50, b=50),
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.warning("Dataset %s unavailable; its panel will be omitted.", name)
        return pd.DataFrame()


def _fig_html(fig: go.Figure, include_js: bool = False) -> str:
    """Render a figure to an embeddable div; the Plotly bundle is included once."""
    return pio.to_html(fig, include_plotlyjs=("inline" if include_js else False),
                       full_html=False, config={"displaylogo": False,
                                                "responsive": True})


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_accuracy_ladder(hierarchy: pd.DataFrame) -> Optional[go.Figure]:
    if hierarchy.empty:
        return None
    frame = hierarchy.sort_values("WAPE")
    colours = [GREEN if v < 15 else AMBER if v < 30 else RED for v in frame["WAPE"]]
    fig = go.Figure(go.Bar(
        x=frame["WAPE"], y=frame["Aggregation_Level"], orientation="h",
        marker_color=colours, text=[f"{v:.2f}%" for v in frame["WAPE"]],
        textposition="outside",
        hovertemplate="%{y}<br>WAPE %{x:.2f}%<extra></extra>"))
    fig.add_vline(x=CONFIG.forecast.portfolio_mape_target, line_dash="dash",
                  line_color=NAVY,
                  annotation_text=f"{CONFIG.forecast.portfolio_mape_target:.0f}% benchmark")
    fig.update_layout(**LAYOUT, title="Forecast Accuracy by Aggregation Level (holdout)",
                      xaxis_title="WAPE (%)", yaxis_title="")
    return fig


def fig_forecast_profile(forecast: pd.DataFrame) -> Optional[go.Figure]:
    if forecast.empty:
        return None
    agg = (forecast.groupby("horizon", as_index=False)
           .agg(Forecast=("Forecast_Units", "sum"), P10=("P10", "sum"),
                P90=("P90", "sum"), P2_5=("P2_5", "sum"), P97_5=("P97_5", "sum")))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P97_5"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P2_5"], fill="tonexty",
                             fillcolor="rgba(46,117,182,0.10)", line=dict(width=0),
                             name="95% interval"))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["P10"], fill="tonexty",
                             fillcolor="rgba(46,117,182,0.24)", line=dict(width=0),
                             name="80% interval"))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["Forecast"], mode="lines+markers",
                             name="Consensus forecast",
                             line=dict(color=NAVY, width=3.5)))
    fig.update_layout(**LAYOUT, title="Portfolio 12-Week Forecast with Prediction Intervals",
                      xaxis_title="Forecast week", yaxis_title="Units",
                      hovermode="x unified")
    return fig


def fig_model_leaderboard(leaderboard: pd.DataFrame) -> Optional[go.Figure]:
    if leaderboard.empty:
        return None
    holdout = leaderboard[leaderboard["Stage"] == "HOLDOUT"]
    if holdout.empty:
        holdout = leaderboard
    fig = go.Figure()
    for index, (tier, group) in enumerate(holdout.groupby("Tier")):
        group = group.sort_values("Median_MAPE")
        fig.add_trace(go.Bar(x=group["Model"], y=group["Median_MAPE"], name=tier,
                             marker_color=[NAVY, BLUE][index % 2],
                             text=[f"{v:.1f}" for v in group["Median_MAPE"]],
                             textposition="outside"))
    fig.update_layout(**LAYOUT, barmode="group",
                      title="Model Competition — Median MAPE on Untouched Holdout",
                      xaxis_title="", yaxis_title="Median MAPE (%)")
    return fig


def fig_champion_mix(accuracy: pd.DataFrame) -> Optional[go.Figure]:
    if accuracy.empty or "Selected_Model" not in accuracy.columns:
        return None
    mix = accuracy["Selected_Model"].value_counts()
    fig = go.Figure(go.Pie(labels=mix.index, values=mix.to_numpy(), hole=0.52,
                           marker=dict(colors=[NAVY, BLUE, GREEN, AMBER, RED,
                                               "#7030A0"][:len(mix)]),
                           textinfo="label+percent"))
    fig.update_layout(**{**LAYOUT, "height": 380},
                      title="Champion Model Selected per SKU (on CV, before holdout)")
    return fig


def fig_feature_importance(importance: pd.DataFrame,
                           shap_df: pd.DataFrame) -> Optional[go.Figure]:
    if importance.empty:
        return None
    top = importance.head(14).sort_values("Importance")
    fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h",
                           marker_color=BLUE,
                           hovertemplate="%{y}<br>Gain %{x:.4f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="XGBoost Feature Importance (gain)",
                      xaxis_title="Gain", yaxis_title="")
    return fig


def fig_shap(shap_df: pd.DataFrame) -> Optional[go.Figure]:
    if shap_df.empty:
        return None
    top = shap_df.head(14).sort_values("Mean_Abs_SHAP")
    fig = go.Figure(go.Bar(x=top["Mean_Abs_SHAP"], y=top["Feature"], orientation="h",
                           marker_color=NAVY))
    fig.update_layout(**LAYOUT, title="SHAP Attribution — Mean |contribution| per Prediction",
                      xaxis_title="Mean |SHAP| (units)", yaxis_title="")
    return fig


def fig_waterfall(waterfall: pd.DataFrame) -> Optional[go.Figure]:
    if waterfall.empty:
        return None
    measures = {"absolute": "absolute", "relative": "relative", "total": "total"}
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=[measures.get(m, "relative") for m in waterfall["Measure"]],
        x=[s.replace(" ", "<br>") for s in waterfall["Step"]],
        y=list(waterfall["Units"]),
        text=[f"{v:,.0f}" for v in waterfall["Units"]],
        textposition="outside",
        connector={"line": {"color": MUTED}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": NAVY}}))
    fig.update_layout(**{**LAYOUT, "height": 460},
                      title="Consensus Bridge: Statistical Baseline → Final Plan",
                      yaxis_title="Units")
    return fig


def fig_scenarios(scenarios: pd.DataFrame) -> Optional[go.Figure]:
    if scenarios.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenarios["Scenario"], y=scenarios["Consensus_Units"],
                         name="Achievable consensus", marker_color=NAVY))
    fig.add_trace(go.Bar(x=scenarios["Scenario"], y=scenarios["Constrained_Out_Units"],
                         name="Constrained out", marker_color=RED))
    fig.update_layout(**LAYOUT, barmode="stack",
                      title="Scenario Comparison — Achievable vs Constrained-Out Volume",
                      xaxis_title="", yaxis_title="Units")
    return fig


def fig_override_distribution(consensus: pd.DataFrame,
                              scenario: str) -> Optional[go.Figure]:
    if consensus.empty:
        return None
    subset = consensus[(consensus["Scenario"] == scenario)
                       & consensus["Has_Commercial_Input"]]
    if subset.empty:
        return None
    threshold = CONFIG.consensus.override_alert_threshold_pct
    fig = go.Figure(go.Histogram(x=subset["Override_Deviation_Pct"], nbinsx=60,
                                 marker_color=BLUE))
    for sign in (1, -1):
        fig.add_vline(x=sign * threshold, line_dash="dash", line_color=RED)
    fig.update_layout(**LAYOUT,
                      title=f"Commercial Override Distribution "
                            f"(governance threshold ±{threshold:.0f}%)",
                      xaxis_title="Deviation from statistical baseline (%)",
                      yaxis_title="SKU-weeks")
    return fig


def fig_capacity_heatmap(capacity: pd.DataFrame) -> Optional[go.Figure]:
    if capacity.empty:
        return None
    supplier = capacity[capacity["Resource_Type"] == "Supplier"]
    if supplier.empty:
        return None
    pivot = supplier.pivot_table(index="Resource_ID", columns="horizon",
                                 values="Utilisation_Pct", aggfunc="max")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index[:22]]
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=[f"W{c}" for c in pivot.columns], y=pivot.index,
        colorscale=[[0, "#E8F3EC"], [0.45, "#FFF3CD"], [0.7, "#F8C9A0"], [1, RED]],
        zmin=0, zmax=200, colorbar=dict(title="Util %"),
        hovertemplate="%{y} · %{x}<br>Utilisation %{z:.0f}%<extra></extra>"))
    fig.update_layout(**{**LAYOUT, "height": 560},
                      title="Supplier Capacity Utilisation by Horizon Week "
                            "(worst 22 suppliers)",
                      xaxis_title="Horizon week", yaxis_title="")
    return fig


def fig_inventory_trajectory(supply_plan: pd.DataFrame) -> Optional[go.Figure]:
    if supply_plan.empty:
        return None
    agg = (supply_plan.groupby("horizon", as_index=False)
           .agg(Demand=("Demand_Units", "sum"), Receipts=("Planned_Receipts", "sum"),
                Closing=("Closing_Inventory", "sum"), Safety=("Safety_Stock", "sum")))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Demand"], name="Demand",
                         marker_color=LIGHT))
    fig.add_trace(go.Bar(x=agg["horizon"], y=agg["Receipts"], name="Planned receipts",
                         marker_color=BLUE, opacity=0.75))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["Closing"],
                             name="Closing inventory", mode="lines+markers",
                             line=dict(color=NAVY, width=3.5)))
    fig.add_trace(go.Scatter(x=agg["horizon"], y=agg["Safety"], name="Safety stock",
                             mode="lines", line=dict(color=RED, width=2, dash="dash")))
    fig.update_layout(**LAYOUT, barmode="group", hovermode="x unified",
                      title="Projected Inventory Trajectory vs Safety Stock",
                      xaxis_title="Horizon week", yaxis_title="Units")
    return fig


def fig_risk_scatter(summary: pd.DataFrame) -> Optional[go.Figure]:
    if summary.empty:
        return None
    plot = summary.copy()
    plot["Min_Cover"] = plot["Min_Cover"].clip(upper=20)
    fig = go.Figure()
    for band in ["Green", "Amber", "Red"]:
        subset = plot[plot["Risk_Band"] == band]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Min_Cover"], y=subset["Supply_Risk_Index"], mode="markers",
            name=band, marker=dict(color=RAG[band], size=8, opacity=0.7,
                                   line=dict(width=0.5, color="white")),
            text=subset[Keys.SKU],
            hovertemplate="%{text}<br>Cover %{x:.1f} wks<br>"
                          "Risk %{y:.0f}<extra></extra>"))
    fig.add_hline(y=CONFIG.supply.risk_red_threshold, line_dash="dash", line_color=RED)
    fig.add_vline(x=CONFIG.supply.target_weeks_of_cover, line_dash="dash",
                  line_color=NAVY)
    fig.update_layout(**{**LAYOUT, "height": 480},
                      title="Weeks of Cover vs Supply Risk Index "
                            "(bottom-right quadrant is the working list)",
                      xaxis_title="Minimum weeks of cover",
                      yaxis_title="Supply Risk Index")
    return fig


def fig_stockout_timing(stockouts: pd.DataFrame) -> Optional[go.Figure]:
    if stockouts.empty:
        return None
    projected = stockouts[stockouts["Event_Type"] == "Projected Stockout"]
    if projected.empty:
        return None
    counts = projected.groupby("horizon").size()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(), marker_color=RED,
                           text=counts.to_numpy(), textposition="outside"))
    fig.update_layout(**LAYOUT,
                      title="Projected Stockout Timing — Weeks of Advance Notice",
                      xaxis_title="Horizon week of projected stockout",
                      yaxis_title="SKU-week events")
    return fig


def fig_spc(spc: pd.DataFrame) -> Optional[go.Figure]:
    if spc.empty:
        return None
    x = spc["Order_Month"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=spc["UCL"], name="UCL (+3σ)", mode="lines",
                             line=dict(color=RED, dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=spc["LCL"], name="LCL (−3σ)", mode="lines",
                             line=dict(color=RED, dash="dash", width=1.5),
                             fill="tonexty", fillcolor="rgba(15,123,79,0.06)"))
    fig.add_trace(go.Scatter(x=x, y=spc["Center_Line"], name="Centre line",
                             mode="lines", line=dict(color=NAVY, width=2)))
    normal = spc[~spc["Special_Cause"].astype(bool)]
    special = spc[spc["Special_Cause"].astype(bool)]
    fig.add_trace(go.Scatter(x=spc["Order_Month"], y=spc["OTIF_Pct"], name="OTIF",
                             mode="lines", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=normal["Order_Month"], y=normal["OTIF_Pct"],
                             name="In control", mode="markers",
                             marker=dict(size=9, color=BLUE)))
    if not special.empty:
        fig.add_trace(go.Scatter(x=special["Order_Month"], y=special["OTIF_Pct"],
                                 name="Special cause", mode="markers",
                                 marker=dict(size=17, color=RED, symbol="x",
                                             line=dict(width=2))))
    fig.add_hline(y=CONFIG.kpi.otif_acceptable, line_dash="dot", line_color=MUTED,
                  annotation_text=f"{CONFIG.kpi.otif_acceptable:.0f}% business threshold")
    fig.update_layout(**{**LAYOUT, "height": 470}, hovermode="x unified",
                      title="Monthly OTIF — Six Sigma Individuals Control Chart "
                            "(σ from moving range)",
                      xaxis_title="Month", yaxis_title="OTIF (%)")
    return fig


def fig_pareto(causes: pd.DataFrame) -> Optional[go.Figure]:
    if causes.empty:
        return None
    frame = causes.sort_values("Value_At_Risk", ascending=False)
    colours = [RED if v else BLUE for v in frame.get("Vital_Few", [False] * len(frame))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame["Root_Cause"], y=frame["Value_At_Risk"],
                         name="Value at risk", marker_color=colours))
    if "Cumulative_Value_Pct" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["Root_Cause"], y=frame["Cumulative_Value_Pct"],
                                 name="Cumulative %", yaxis="y2", mode="lines+markers",
                                 line=dict(color=NAVY, width=3)))
    fig.update_layout(**{**LAYOUT, "height": 470},
                      title="Root Cause Pareto — Ranked by Value at Risk, not Frequency",
                      xaxis_title="",
                      yaxis_title=f"Value at risk ({RC.currency_symbol})",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                  title="Cumulative %"))
    return fig


def fig_otif_dimension(frame: pd.DataFrame, key: str, title: str
                       ) -> Optional[go.Figure]:
    if frame.empty or key not in frame.columns:
        return None
    ordered = frame.sort_values("OTIF_Pct")
    colours = [RAG.get(b, BLUE) for b in ordered["Benchmark"]]
    fig = go.Figure(go.Bar(x=ordered["OTIF_Pct"], y=ordered[key], orientation="h",
                           marker_color=colours,
                           text=[f"{v:.1f}%" for v in ordered["OTIF_Pct"]],
                           textposition="outside",
                           customdata=ordered[["Orders"]],
                           hovertemplate="%{y}<br>OTIF %{x:.1f}%<br>"
                                         "Orders %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=CONFIG.kpi.otif_acceptable, line_dash="dash", line_color=NAVY)
    fig.update_layout(**LAYOUT, title=title, xaxis_title="OTIF (%)", yaxis_title="")
    return fig


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _kpi_card(label: str, value: str, sub: str, status: str = "neutral") -> str:
    colour = {"good": GREEN, "warn": AMBER, "bad": RED}.get(status, NAVY)
    return f"""
    <div class="kpi" style="border-top-color:{colour}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{colour}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


def _table(df: pd.DataFrame, table_id: str, max_rows: int = 150,
           highlight_col: Optional[str] = None) -> str:
    """Searchable, sortable HTML table (sorting handled by inline JS)."""
    if df is None or df.empty:
        return '<p class="muted">No records met the criteria for this analysis.</p>'
    frame = df.head(max_rows).copy()
    for column in frame.select_dtypes("float").columns:
        frame[column] = frame[column].round(2)

    header = "".join(f"<th onclick=\"sortTable('{table_id}',{i})\">"
                     f"{c.replace('_', ' ')}</th>"
                     for i, c in enumerate(frame.columns))
    rows = []
    for _, record in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = record[column]
            style = ""
            if highlight_col and column == highlight_col:
                colour = RAG.get(str(value))
                if colour:
                    style = (f' style="background:{colour};color:white;'
                             f'font-weight:600;text-align:center"')
            text = "" if pd.isna(value) else str(value)
            cells.append(f"<td{style}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    note = (f'<p class="muted">Showing {len(frame):,} of {len(df):,} rows. '
            f'Full data in the CSV exports.</p>' if len(df) > max_rows else "")
    return f"""
    <input class="search" onkeyup="filterTable('{table_id}',this.value)"
           placeholder="Filter {len(df):,} rows…">
    <div class="table-wrap">
      <table id="{table_id}"><thead><tr>{header}</tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
    </div>{note}"""


def _insight_cards(insights: pd.DataFrame) -> str:
    if insights.empty:
        return '<p class="muted">No insights generated.</p>'
    blocks = []
    for severity in ["Critical", "High", "Medium", "Info"]:
        subset = insights[insights["Severity"] == severity]
        if subset.empty:
            continue
        items = []
        for _, row in subset.iterrows():
            action = row.get("Recommended_Action", "")
            action_html = (f'<div class="ins-action">→ {action}</div>'
                           if isinstance(action, str) and action.strip() else "")
            items.append(f"""
            <div class="insight" style="border-left-color:{RAG.get(severity, BLUE)}">
              <div>
                <span class="pill" style="background:{RAG.get(severity, BLUE)}">
                  {row['Domain']}</span>
                <span class="owner">Owner: {row.get('Owner', '—')}</span>
              </div>
              <div class="ins-head">{row['Headline']}</div>
              <div class="ins-detail">{row['Detail']}</div>
              {action_html}
            </div>""")
        blocks.append(f'<h3>{severity} findings ({len(subset)})</h3>'
                      + "".join(items))
    return "".join(blocks)


def _panel(title: str, body: str, subtitle: str = "") -> str:
    sub = f'<div class="panel-sub">{subtitle}</div>' if subtitle else ""
    return f'<section class="panel"><h2>{title}</h2>{sub}{body}</section>'


CSS = """
*{box-sizing:border-box}
body{margin:0;font-family:Inter,'Segoe UI',Helvetica,Arial,sans-serif;
     background:#F4F6FA;color:#1B1F2A}
header{background:#1F3864;color:#fff;padding:20px 30px;display:flex;
       justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
header h1{margin:0;font-size:20px;letter-spacing:.04em}
header .sub{font-size:11px;color:#B9C6DE;margin-top:3px}
.badge{padding:7px 15px;border-radius:20px;font-size:11px;font-weight:700;
       letter-spacing:.05em}
nav{background:#fff;display:flex;gap:2px;padding:0 22px;overflow-x:auto;
    border-bottom:1px solid #E3E8F0;position:sticky;top:0;z-index:20}
nav button{background:none;border:none;padding:15px 20px;font-size:13px;
           font-weight:600;color:#6B7280;cursor:pointer;border-bottom:3px solid transparent;
           white-space:nowrap;font-family:inherit}
nav button:hover{color:#1F3864;background:#F8FAFD}
nav button.active{color:#1F3864;border-bottom-color:#2E75B6}
main{padding:22px 30px 60px}
.tab{display:none}.tab.active{display:block}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.kpi{background:#fff;border-radius:10px;padding:14px 17px;flex:1;min-width:165px;
     border-top:3px solid #1F3864;box-shadow:0 1px 3px rgba(16,24,40,.08)}
.kpi-label{font-size:10.5px;color:#6B7280;text-transform:uppercase;
           letter-spacing:.06em;font-weight:600}
.kpi-value{font-size:25px;font-weight:700;margin-top:5px;line-height:1.1}
.kpi-sub{font-size:11px;color:#6B7280;margin-top:3px}
.panel{background:#fff;border-radius:10px;padding:18px;margin-bottom:16px;
       box-shadow:0 1px 3px rgba(16,24,40,.08)}
.panel h2{margin:0 0 4px;font-size:14.5px;color:#1F3864}
.panel-sub{font-size:11.5px;color:#6B7280;margin-bottom:12px;line-height:1.55}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}
.table-wrap{overflow-x:auto;max-height:460px;overflow-y:auto;border:1px solid #E3E8F0;
            border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:11.5px}
th{background:#1F3864;color:#fff;padding:9px 10px;text-align:left;font-size:10.5px;
   text-transform:uppercase;letter-spacing:.03em;cursor:pointer;position:sticky;top:0;
   white-space:nowrap}
th:hover{background:#2E75B6}
td{padding:7px 10px;border-bottom:1px solid #EEF1F6}
tbody tr:nth-child(even){background:#F8FAFC}
tbody tr:hover{background:#EEF4FC}
.search{width:100%;padding:9px 12px;margin-bottom:10px;border:1px solid #D8DFEA;
        border-radius:6px;font-size:12.5px;font-family:inherit}
.muted{color:#6B7280;font-size:11.5px;font-style:italic}
.insight{padding:13px 15px;border-left:3px solid #2E75B6;background:#FBFCFE;
         margin-bottom:10px;border-radius:0 8px 8px 0}
.pill{color:#fff;padding:2px 9px;border-radius:10px;font-size:10px;font-weight:700;
      letter-spacing:.04em}
.owner{font-size:11px;color:#6B7280;margin-left:9px}
.ins-head{font-size:13px;font-weight:600;margin:7px 0 4px}
.ins-detail{font-size:12px;color:#6B7280;line-height:1.55}
.ins-action{font-size:12px;color:#1F3864;margin-top:6px;font-weight:600}
h3{font-size:13px;color:#1F3864;margin:18px 0 9px}
footer{padding:22px 30px;color:#6B7280;font-size:11px;text-align:center;
       border-top:1px solid #E3E8F0;background:#fff;line-height:1.7}
"""

JS = """
function showTab(id, btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}
function filterTable(id, q){
  q = q.toLowerCase();
  document.querySelectorAll('#'+id+' tbody tr').forEach(r=>{
    r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
  });
}
function sortTable(id, col){
  const tb = document.querySelector('#'+id+' tbody');
  const rows = Array.from(tb.rows);
  const dir = tb.dataset.dir === String(col) ? -1 : 1;
  tb.dataset.dir = dir === 1 ? String(col) : '';
  rows.sort((a,b)=>{
    const x = a.cells[col].innerText.replace(/[,%₹]/g,'').trim();
    const y = b.cells[col].innerText.replace(/[,%₹]/g,'').trim();
    const nx = parseFloat(x), ny = parseFloat(y);
    if(!isNaN(nx) && !isNaN(ny)) return (nx-ny)*dir;
    return x.localeCompare(y)*dir;
  });
  rows.forEach(r=>tb.appendChild(r));
}
"""


def build_static_dashboard(output_path: Optional[Path] = None) -> Path:
    """Assemble the complete single-file dashboard."""
    output_path = output_path or (OUTPUT_DIR / "sop_dashboard.html")

    # -- load published data ------------------------------------------------ #
    forecast = _read("demand_forecast.csv")
    hierarchy = _read("forecast_accuracy_hierarchy.csv")
    accuracy = _read("forecast_sku_accuracy.csv")
    leaderboard = _read("model_leaderboard.csv")
    importance = _read("feature_importance.csv")
    shap_df = _read("shap_summary.csv")
    consensus = _read("consensus_plan.csv")
    override_log = _read("override_log.csv")
    scenarios = _read("scenario_summary.csv")
    waterfall = _read("consensus_waterfall.csv")
    supply_plan = _read("supply_plan.csv")
    supply_summary = _read("supply_sku_summary.csv")
    capacity = _read("capacity_alerts.csv")
    stockouts = _read("stockout_risks.csv")
    spc = _read("spc_chart.csv")
    violations = _read("spc_violations.csv")
    causes = _read("root_cause_analysis.csv")
    scorecard = _read("supplier_scorecard.csv")
    otif_cat = _read("otif_by_category.csv")
    otif_chan = _read("otif_by_channel.csv")
    otif_month = _read("otif_by_month.csv")
    insights = _read("executive_insights.csv")

    manifest: Dict = {}
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    scenario = manifest.get("scenario", CONFIG.consensus.default_scenario)
    currency = RC.currency_symbol

    # -- headline KPIs ------------------------------------------------------ #
    review_wape = None
    if not hierarchy.empty:
        match = hierarchy[hierarchy["Aggregation_Level"] == "Category / Month"]
        if not match.empty:
            review_wape = float(match.iloc[0]["WAPE"])
    sku_week = (hierarchy[hierarchy["Aggregation_Level"] == "SKU / Week"]
                if not hierarchy.empty else pd.DataFrame())

    service = manifest.get("service", {})
    supply_diag = manifest.get("supply", {})
    consensus_diag = manifest.get("consensus", {})
    otif = service.get("overall_otif_pct")
    critical = int((insights["Severity"] == "Critical").sum()) if not insights.empty else 0

    overview_kpis = "".join([
        _kpi_card("Review-Level WAPE",
                  f"{review_wape:.2f}%" if review_wape else "n/a",
                  "Category / Month (S&OP sign-off)",
                  "good" if review_wape and review_wape < 15 else "warn"),
        _kpi_card("Forecast Value Added",
                  f"+{float(sku_week.iloc[0]['FVA_vs_Naive']):.1f} pts"
                  if not sku_week.empty else "n/a",
                  "vs seasonal-naive benchmark", "good"),
        _kpi_card("OTIF", f"{otif:.2f}%" if otif else "n/a",
                  service.get("benchmark", ""),
                  "good" if otif and otif >= 95 else "warn" if otif and otif >= 90 else "bad"),
        _kpi_card("Consensus Value",
                  human_number(consensus_diag.get("consensus_value", 0), currency),
                  f"{scenario} scenario"),
        _kpi_card("Revenue At Risk",
                  human_number(supply_diag.get("revenue_at_risk", 0), currency),
                  "projected stockouts", "bad"),
        _kpi_card("Red-Risk SKUs",
                  f"{supply_diag.get('risk_distribution', {}).get('Red', 0):,}",
                  f"of {supply_diag.get('skus_planned', 0):,} planned", "warn"),
    ])

    supply_kpis = "".join([
        _kpi_card("Capacity Breaches", f"{supply_diag.get('capacity_breaches', 0):,}",
                  "resource-weeks above 100%", "bad"),
        _kpi_card("Expedite Required", f"{supply_diag.get('expedite_required', 0):,}",
                  "stockout inside lead time", "bad"),
        _kpi_card("Projected Stockouts", f"{supply_diag.get('skus_with_stockout', 0):,}",
                  f"earliest week {supply_diag.get('earliest_stockout_week', '—')}", "warn"),
        _kpi_card("Working Capital",
                  human_number(supply_diag.get("total_working_capital", 0), currency),
                  "week 1 inventory value"),
        _kpi_card("Constrained Out",
                  human_number(consensus_diag.get("constrained_out_value", 0), currency),
                  "demand capacity cannot serve", "warn"),
    ])

    service_kpis = "".join([
        _kpi_card("OTIF", f"{otif:.2f}%" if otif else "n/a",
                  service.get("benchmark", ""),
                  "good" if otif and otif >= 95 else "warn" if otif and otif >= 90 else "bad"),
        _kpi_card("On Time", f"{service.get('on_time_pct', 0):.2f}%", "by promised date"),
        _kpi_card("In Full", f"{service.get('in_full_pct', 0):.2f}%", "≥98% of ordered qty"),
        _kpi_card("Special-Cause Months", f"{service.get('special_cause_months', 0)}",
                  "SPC signal, not noise",
                  "bad" if service.get("special_cause_months", 0) else "good"),
        _kpi_card("Top Root Cause", str(service.get("top_root_cause", "—")),
                  f"{service.get('top_root_cause_share', 0):.0f}% of failures", "warn"),
    ])

    # -- figures (first one carries the Plotly bundle) ---------------------- #
    figures: List[tuple] = [
        ("overview", fig_accuracy_ladder(hierarchy)),
        ("overview", fig_forecast_profile(forecast)),
        ("forecast", fig_model_leaderboard(leaderboard)),
        ("forecast", fig_champion_mix(accuracy)),
        ("forecast", fig_feature_importance(importance, shap_df)),
        ("forecast", fig_shap(shap_df)),
        ("consensus", fig_waterfall(waterfall)),
        ("consensus", fig_scenarios(scenarios)),
        ("consensus", fig_override_distribution(consensus, scenario)),
        ("supply", fig_capacity_heatmap(capacity)),
        ("supply", fig_inventory_trajectory(supply_plan)),
        ("supply", fig_risk_scatter(supply_summary)),
        ("supply", fig_stockout_timing(stockouts)),
        ("service", fig_spc(spc)),
        ("service", fig_pareto(causes)),
        ("service", fig_otif_dimension(otif_cat, Keys.CATEGORY, "OTIF by Category")),
        ("service", fig_otif_dimension(otif_chan, Keys.CHANNEL, "OTIF by Channel")),
    ]
    rendered: Dict[str, List[str]] = {}
    first = True
    for tab, fig in figures:
        if fig is None:
            continue
        rendered.setdefault(tab, []).append(_fig_html(fig, include_js=first))
        first = False

    def figs(tab: str, index: Optional[int] = None) -> str:
        items = rendered.get(tab, [])
        if index is None:
            return "".join(items)
        return items[index] if index < len(items) else ""

    generated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

    # -- tab bodies --------------------------------------------------------- #
    tab_overview = f"""
    <div class="kpis">{overview_kpis}</div>
    {_panel("Forecast accuracy by aggregation level", figs("overview", 0),
            "Forecast error cancels as demand aggregates. A single accuracy number "
            "without its aggregation level is meaningless — this ladder is the honest "
            "answer. Measured on a 12-week holdout excluded from model selection and "
            "ensemble weighting.")}
    {_panel("Portfolio forecast with prediction intervals", figs("overview", 1),
            "Intervals are calibrated on cross-validation residuals; the horizon "
            "variance growth parameter is fitted from observed data, not assumed. "
            "Empirical coverage is measured on the holdout and reported as measured.")}
    {_panel("Executive findings", _insight_cards(insights.head(8)) if not insights.empty
            else '<p class="muted">No findings.</p>',
            "Generated from calculated metrics crossing configured thresholds. "
            "A clean result produces no finding rather than a reassuring one.")}
    """

    tab_forecast = f"""
    {_panel("Model competition on the untouched holdout", figs("forecast", 0),
            "Four model families plus an inverse-MAPE ensemble and a seasonal-naive "
            "benchmark. SARIMA and Prophet run on a revenue-and-category stratified "
            "panel; results are reported as panel results and never extrapolated.")}
    <div class="grid2">
      {_panel("Champion model per SKU", figs("forecast", 1),
              "Selected on cross-validation performance before the holdout is opened. "
              "The ensemble competes as one candidate and does not win by default.")}
      {_panel("XGBoost feature importance", figs("forecast", 2),
              "Gain-based importance from the global direct multi-horizon model.")}
    </div>
    {_panel("SHAP attribution", figs("forecast", 3),
            "Mean absolute SHAP value — the average contribution each feature makes "
            "to a single prediction.")}
    {_panel("Accuracy detail by SKU", _table(
        accuracy.sort_values("MAPE") if not accuracy.empty else accuracy,
        "tbl-acc", 200, "Accuracy_Band"))}
    """

    tab_consensus = f"""
    {_panel("Consensus bridge", figs("consensus", 0),
            "The statistical baseline is never overwritten — it is carried through "
            "every stage as its own column, so when the plan misses you can establish "
            "whether the model was wrong or the override was.")}
    <div class="grid2">
      {_panel("Scenario comparison", figs("consensus", 1),
              "Scenario choice is a working-capital and service decision, not a "
              "forecasting one.")}
      {_panel("Override distribution vs governance threshold", figs("consensus", 2),
              "Deviations beyond ±20% require justification; beyond ±45% require "
              "S&OP board sign-off.")}
    </div>
    {_panel("Override audit log", _table(
        override_log[override_log["Override_Alert"]] if not override_log.empty
        and "Override_Alert" in override_log.columns else override_log,
        "tbl-ovr", 200, "Governance_Status"),
        "Every override carries its driver, justification, submitter, governance "
        "status and a UTC audit timestamp. Flagged overrides without a valid "
        "justification are BLOCKED.")}
    """

    tab_supply = f"""
    <div class="kpis">{supply_kpis}</div>
    {_panel("Supplier capacity utilisation heatmap", figs("supply", 0),
            "Capacity breaches exist only in aggregate. A single SKU asking for 3,000 "
            "units is never a problem; forty SKUs on the same supplier line asking for "
            "3,000 each in the same week is.")}
    {_panel("Projected inventory trajectory", figs("supply", 1))}
    <div class="grid2">
      {_panel("Weeks of cover vs supply risk", figs("supply", 2),
              "The bottom-right quadrant is where a planner should spend the morning.")}
      {_panel("Stockout timing", figs("supply", 3),
              "How much advance notice the plan gives before each shortage.")}
    </div>
    {_panel("Capacity constraint alert log", _table(
        capacity[capacity["Severity"] != "OK"] if not capacity.empty else capacity,
        "tbl-cap", 200, "Severity"))}
    {_panel("SKU risk register", _table(supply_summary, "tbl-risk", 200, "Risk_Band"),
            "Supply Risk Index blends stockout proximity, capacity utilisation, cover "
            "shortfall, lead-time exposure and forecast uncertainty using documented "
            "weights. Every component is inspectable per SKU.")}
    """

    tab_service = f"""
    <div class="kpis">{service_kpis}</div>
    {_panel("Six Sigma individuals control chart", figs("service", 0),
            "Sigma is estimated from the average moving range (MR̄ / 1.128), not the "
            "sample standard deviation, which would be inflated by the very special "
            "causes the chart exists to detect. Points inside the limits are routine "
            "variation and must not be reacted to — doing so adds variation rather "
            "than removing it.")}
    {_panel("Western Electric rule violations", _table(violations, "tbl-we", 60)
            if not violations.empty else
            '<p style="color:#0F7B4F;font-weight:600">No Western Electric rule '
            'violations — the process is statistically stable.</p>')}
    {_panel("Root cause Pareto", figs("service", 1),
            "Ranked by value at risk rather than frequency: the most common failure "
            "mode is rarely the most expensive one.")}
    <div class="grid2">
      {_panel("OTIF by category", figs("service", 2))}
      {_panel("OTIF by channel", figs("service", 3))}
    </div>
    {_panel("Supplier scorecard", _table(scorecard, "tbl-sup", 200, "Benchmark"),
            "Every cut carries an order count. A supplier at 100% on three orders and "
            "one at 94% on four hundred are not comparable.")}
    """

    tab_insights = _panel(
        "Full insight register", _insight_cards(insights),
        "Each finding carries a severity, quantified impact, accountable owner and "
        "recommended action. An insight with no owner is a fact, not an insight.")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S&amp;OP Intelligence System — Command Centre</title>
<style>{CSS}</style></head><body>
<header>
  <div>
    <h1>S&amp;OP INTELLIGENCE SYSTEM</h1>
    <div class="sub">Demand &amp; Supply Planning Command Centre &nbsp;·&nbsp;
      {scenario} scenario &nbsp;·&nbsp; Generated {generated}</div>
  </div>
  <div class="badge" style="background:{RED if critical else GREEN}">
    {f'{critical} CRITICAL ALERTS' if critical else 'NO CRITICAL ALERTS'}</div>
</header>
<nav>
  <button class="active" onclick="showTab('t-overview',this)">Overview</button>
  <button onclick="showTab('t-forecast',this)">1 · Forecast &amp; ML</button>
  <button onclick="showTab('t-consensus',this)">2 · Consensus Plan</button>
  <button onclick="showTab('t-supply',this)">3 · Supply Constraints</button>
  <button onclick="showTab('t-service',this)">4 · OTIF &amp; SPC</button>
  <button onclick="showTab('t-insights',this)">5 · Insights</button>
</nav>
<main>
  <div id="t-overview" class="tab active">{tab_overview}</div>
  <div id="t-forecast" class="tab">{tab_forecast}</div>
  <div id="t-consensus" class="tab">{tab_consensus}</div>
  <div id="t-supply" class="tab">{tab_supply}</div>
  <div id="t-service" class="tab">{tab_service}</div>
  <div id="t-insights" class="tab">{tab_insights}</div>
</main>
<footer>
  <strong>All data is synthetic and simulated for portfolio demonstration.</strong>
  It does not originate from, and does not represent, any employer, client or real
  organisation.<br>
  Generated by the S&amp;OP Intelligence System · Output 2 of the Supply Chain
  Command Center · Every figure derives from the same published datasets as the
  Excel pack.
</footer>
<script>{JS}</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Static dashboard written: %s (%.1f MB, %d charts, offline-capable).",
                output_path.name, size_mb, sum(len(v) for v in rendered.values()))
    return output_path


if __name__ == "__main__":
    build_static_dashboard()
