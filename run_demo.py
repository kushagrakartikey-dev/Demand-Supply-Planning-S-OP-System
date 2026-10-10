#!/usr/bin/env python
"""
One-command runner for the S&OP Intelligence System.

    python run_demo.py                    # full pipeline + launch dashboard
    python run_demo.py --export-reports   # pipeline + Excel/CSV, no dashboard
    python run_demo.py --open-dashboard   # launch the dashboard (runs pipeline if needed)
    python run_demo.py --quick            # reduced scale for a fast smoke test
    python run_demo.py --scenario Aggressive
    python run_demo.py --no-dashboard --skip-tests

Every stage runs from a single seeded configuration, so two runs on the same
machine produce identical numbers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import webbrowser
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planning.config import OUTPUT_DIR, PROCESSED_DIR, load_config
from src.shared.utils import get_logger, human_number

logger = get_logger("run_demo")

BANNER = r"""
================================================================================
   S&OP INTELLIGENCE SYSTEM
   Demand & Supply Planning Decision-Support Platform
   Output 2 of the Supply Chain Command Center

   Data: synthetic / simulated for portfolio demonstration
================================================================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_demo.py",
        description="Run the complete S&OP intelligence pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python run_demo.py
              python run_demo.py --export-reports
              python run_demo.py --open-dashboard
              python run_demo.py --quick --no-dashboard
        """))
    parser.add_argument("--export-reports", action="store_true",
                        help="Run the pipeline and write all Excel/CSV outputs.")
    parser.add_argument("--open-dashboard", action="store_true",
                        help="Launch the dashboard and open a browser tab.")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip the dashboard entirely.")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Reuse existing outputs and go straight to the dashboard.")
    parser.add_argument("--reuse-data", action="store_true",
                        help="Reuse the dataset in data/raw instead of regenerating it.")
    parser.add_argument("--reuse-forecast", action="store_true",
                        help="Reuse the cached forecast in outputs/cache instead of "
                             "refitting. Forecasting is the expensive stage; this makes "
                             "downstream iteration fast.")
    parser.add_argument("--forecast-only", action="store_true",
                        help="Run data generation and forecasting only, then cache and "
                             "stop. Useful on constrained hardware.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced scale (120 SKUs, 12-SKU panel, 1 CV fold).")
    parser.add_argument("--scenario", default=None,
                        choices=["Conservative", "Balanced", "Aggressive"],
                        help="Planning scenario for the published plan.")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run the pytest suite after the pipeline.")
    parser.add_argument("--split-workbooks", action="store_true",
                        help="Also emit the eight per-domain Excel files alongside "
                             "the consolidated master workbook.")
    parser.add_argument("--full-master-data", action="store_true",
                        help="Write every row to the Master Data sheet instead "
                             "of a 25-row sample. Larger workbook.")
    parser.add_argument("--static-dashboard", action="store_true",
                        help="Also export a single-file offline HTML dashboard to "
                             "outputs/sop_dashboard.html (no server required).")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port.")
    parser.add_argument("--host", type=str, default=None, help="Dashboard host.")
    return parser.parse_args()


def apply_quick_mode() -> None:
    """Shrink the workload for a fast end-to-end smoke test."""
    os.environ.setdefault("SOP_N_SKUS", "120")
    os.environ.setdefault("SOP_N_ORDERS", "2000")
    os.environ.setdefault("SOP_PANEL_SIZE", "12")
    os.environ.setdefault("SOP_CV_FOLDS", "1")
    logger.info("QUICK MODE: 120 SKUs, 12-SKU competition panel, 1 CV fold.")


def print_report(result) -> None:
    """Print the build completion report to stdout."""
    m = result.manifest
    fc, cons, sup, svc = m["forecast"], m["consensus"], m["supply"], m["service"]
    currency = load_config().report.currency_symbol

    def line(label: str, value: object) -> str:
        return f"  {label:<42} {value}"

    print("\n" + "=" * 80)
    print("  BUILD COMPLETION REPORT")
    print("=" * 80)

    print("\nDATA")
    for entry in m["datasets"]:
        print(line(entry["dataset"], f"{entry['rows']:,} rows x {entry['columns']} cols"))

    print("\nFORECASTING")
    print(line("SKUs forecast", f"{fc['skus']:,}"))
    print(line("History", f"{fc['weeks_history']} weeks"))
    print(line("Models implemented", ", ".join(fc["models"])))
    print(line("Rolling-origin CV folds", fc["cv_folds"]))
    print(line("Four-model competition panel", f"{fc['competition_panel']} SKUs"))
    print(line("Champion model mix", fc["champion_mix"]))
    print(line("Holdout median SKU MAPE",
               f"{fc['holdout_median_sku_mape']:.1f}%"
               if fc["holdout_median_sku_mape"] else "n/a"))
    print(line("Holdout WAPE (Category/Month)",
               f"{fc['review_level_wape']:.1f}%" if fc["review_level_wape"] else "n/a"))
    coverage = fc.get("interval_coverage", {})
    print(line("Prediction interval coverage",
               f"80% band -> {coverage.get('coverage_80', float('nan')):.1f}%, "
               f"95% band -> {coverage.get('coverage_95', float('nan')):.1f}%"))

    print("\n  Accuracy by aggregation level (untouched holdout):")
    for row in fc.get("accuracy_by_level", []):
        flag = "PASS" if row["Meets_15pct_Benchmark"] else "    "
        print(f"    {flag}  {row['Aggregation_Level']:<28} "
              f"WAPE {row['WAPE']:>6.2f}%   MAPE {row['MAPE']:>6.2f}%   "
              f"FVA {row.get('FVA_vs_Naive', 0):+.1f}pts")

    print("\nCONSENSUS PLANNING")
    print(line("Scenario", cons["scenario"]))
    print(line("Consensus volume", f"{cons['consensus_units']:,.0f} units"))
    print(line("Consensus value", human_number(cons["consensus_value"], currency)))
    print(line("SKUs with commercial override", f"{cons['skus_with_commercial_input']:,}"))
    print(line("Override alerts (>20% deviation)", f"{cons['override_alerts']:,}"))
    print(line("Board escalations", f"{cons['board_escalations']:,}"))
    print(line("Demand constrained out",
               f"{cons['constrained_out_units']:,.0f} units "
               f"({human_number(cons['constrained_out_value'], currency)})"))

    print("\nSUPPLY PLANNING")
    print(line("SKUs planned", f"{sup['skus_planned']:,}"))
    print(line("Risk distribution", sup["risk_distribution"]))
    print(line("SKUs with projected stockout", f"{sup['skus_with_stockout']:,}"))
    print(line("Earliest stockout", f"week {sup['earliest_stockout_week']}"
               if sup["earliest_stockout_week"] else "none in horizon"))
    print(line("Requiring expedite (inside lead time)", f"{sup['expedite_required']:,}"))
    print(line("Capacity breaches / warnings",
               f"{sup['capacity_breaches']:,} / {sup['capacity_warnings']:,}"))
    print(line("Revenue at risk", human_number(sup["revenue_at_risk"], currency)))
    print(line("Working capital (week 1)",
               human_number(sup["total_working_capital"], currency)))

    print("\nSERVICE (OTIF & SPC)")
    print(line("Orders analysed", f"{svc['total_orders']:,}"))
    print(line("OTIF", f"{svc['overall_otif_pct']:.2f}%  ({svc['benchmark']})"))
    print(line("On-time / In-full",
               f"{svc['on_time_pct']:.2f}% / {svc['in_full_pct']:.2f}%"))
    print(line("Control limits",
               f"[{svc['lcl']:.2f}%, {svc['ucl']:.2f}%] "
               f"around {svc['center_line']:.2f}%"))
    print(line("Special-cause months", svc["special_cause_months"]))
    print(line("Western Electric violations", svc["we_rule_violations"]))
    print(line("Top root cause",
               f"{svc['top_root_cause']} ({svc['top_root_cause_share']:.1f}% of failures)"))

    print("\nINSIGHTS")
    print(line("Generated", f"{m['insights']['total']} findings"))
    print(line("By severity", m["insights"]["by_severity"]))
    if not result.insights.empty:
        print("\n  Top findings:")
        for _, row in result.insights.head(5).iterrows():
            print(f"    [{row['Severity']:>8}] {row['Headline']}")

    print("\nOUTPUTS")
    print(line("WORKBOOK",
               "outputs/SOP_Command_Center.xlsx  (9 sheets, chart-led)"))
    print(line("CSV exports", f"{m['outputs']['csv_exports']} files in data/processed/"))
    if m["outputs"].get("tableau_tables"):
        print(line("TABLEAU PACK",
                   f"outputs/tableau/  ({m['outputs']['tableau_tables']} tables, "
                   f"{m['outputs']['tableau_rows']:,} rows, .hyper extract)"))
    print(line("Run manifest", "outputs/run_manifest.json"))
    print(line("Runtime", f"{m['runtime_seconds']:.1f}s"))
    print("=" * 80 + "\n")


def run_tests() -> bool:
    print("\nRunning test suite...\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                             "--tb=short"], cwd=ROOT)
    return result.returncode == 0


def launch_dashboard(host: Optional[str], port: Optional[int], open_browser: bool) -> None:
    config = load_config()
    host = host or config.report.dashboard_host
    port = port or config.report.dashboard_port
    url = f"http://{host}:{port}"

    print("\n" + "=" * 80)
    print(f"  DASHBOARD LIVE AT  {url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 80 + "\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    sys.path.insert(0, str(ROOT / "dashboard"))
    from dashboard.dashboard import run_dashboard
    try:
        run_dashboard(host=host, port=port)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


def main() -> int:
    args = parse_args()
    print(BANNER)

    if args.quick:
        apply_quick_mode()

    if not args.skip_pipeline:
        from src.planning.pipeline import run_pipeline
        try:
            result = run_pipeline(
                config=load_config(),
                scenario=args.scenario,
                regenerate_data=not args.reuse_data,
                export_reports=True,
                reuse_forecast=args.reuse_forecast,
                forecast_only=args.forecast_only,
                split_workbooks=args.split_workbooks,
                full_master_data=args.full_master_data)
        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            return 1
        if args.forecast_only:
            print("\nForecast stage complete and cached. Continue with:\n"
                  "  python run_demo.py --reuse-data --reuse-forecast\n")
            return 0
        print_report(result)

        if args.static_dashboard or args.export_reports:
            try:
                from dashboard.static_export import build_static_dashboard
                path = build_static_dashboard()
                print(f"Offline dashboard: {path}\n"
                      f"  Open it directly in any browser -- no server needed.\n")
            except Exception as exc:
                logger.warning("Static dashboard export failed: %s", exc)
    else:
        if not (PROCESSED_DIR / "demand_forecast.csv").exists():
            logger.error("--skip-pipeline was used but no processed data exists. "
                         "Run `python run_demo.py` first.")
            return 1
        logger.info("Skipping pipeline; using existing outputs.")

    if args.run_tests and not run_tests():
        logger.warning("Test suite reported failures.")

    launch = not args.no_dashboard and not args.export_reports
    if args.open_dashboard:
        launch = True
    if launch:
        launch_dashboard(args.host, args.port, open_browser=args.open_dashboard)
    else:
        config = load_config()
        print(f"Dashboard not launched. Start it with:\n"
              f"  python run_demo.py --open-dashboard\n"
              f"  python dashboard/dashboard.py   "
              f"(http://{config.report.dashboard_host}:{config.report.dashboard_port})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
