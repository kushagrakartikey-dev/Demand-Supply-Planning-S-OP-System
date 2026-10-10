# S&OP Intelligence System

**An end-to-end, AI/ML-enabled Sales & Operations Planning decision-support platform for FMCG, modern retail, e-commerce and 3PL.**

Replaces a manual Excel/PowerPoint S&OP cycle with an auditable, automated workflow:

```
Synthetic Supply Chain Data → Demand Forecasting (4 models) → Model Competition
→ Ensemble → Consensus Planning → MRP Supply Planning → Constraint Detection
→ Inventory Projection → OTIF / Six Sigma SPC → Executive Excel Pack → Dashboard
```

> **Output 2 of 6** in a Supply Chain Command Center portfolio. Runs standalone; designed to integrate with Output 1 (Strategic Sourcing) and Output 5 (Inventory Optimisation) through shared keys and published data contracts.

> ⚠️ **All data is synthetic**, generated programmatically for portfolio demonstration. It does not originate from and does not represent any employer, client or real organisation.

**[▶ Open the live dashboard](https://kushagrakartikey-dev.github.io/Demand-Supply-Planning-S-OP-System/)** — 17 charts, six tabs, no install required.

---

## Quickstart

```bash
git clone https://github.com/kushagrakartikey-dev/Demand-Supply-Planning-S-OP-System.git
cd Demand-Supply-Planning-S-OP-System
pip install -r requirements.txt
python run_demo.py
```

> Requires Python 3.10 or 3.11. On macOS, `xgboost` needs the OpenMP runtime: `brew install libomp`.

One command generates the data, trains and compares four forecasting models, builds the consensus and supply plans, runs the OTIF/SPC analysis, writes a nine-sheet Excel workbook and 31 CSVs, and launches the dashboard at `http://127.0.0.1:8050`.

| Command | Effect |
|---|---|
| `python run_demo.py` | Full pipeline, then launch dashboard |
| `python run_demo.py --export-reports` | Pipeline + all outputs, no dashboard |
| `python run_demo.py --open-dashboard` | Launch dashboard, open browser |
| `python run_demo.py --quick` | Reduced scale (120 SKUs) — fast smoke test |
| `python run_demo.py --scenario Aggressive` | Run a different planning scenario |
| `python run_demo.py --skip-pipeline` | Dashboard only, reuse existing outputs |
| `python run_demo.py --forecast-only` | Run and cache the expensive forecast stage, then stop |
| `python run_demo.py --reuse-data --reuse-forecast` | Skip refitting; iterate on downstream logic in ~80s |
| `python run_demo.py --static-dashboard` | Also export the offline single-file HTML dashboard |
| `python dashboard/static_export.py` | Rebuild the offline dashboard from existing outputs |
| `python run_demo.py --full-master-data` | Write every row to the Master Data sheet instead of a 25-row sample |
| `python scripts/verify_outputs.py` | Verify the published artefacts — chart XML, hyperlinks, formulas, `.hyper` |
| `pytest tests/ -q` | Run the 127-test suite |

**Runtime (measured, Apple Silicon, single run from clean):** 161s at full scale (500 SKUs); 62s with `--quick`. Re-running downstream logic with `--reuse-data --reuse-forecast` takes ~25s.

Forecasting is the expensive stage and is **checkpointed per cutoff** to `outputs/cache/stages/`. An interrupted run resumes rather than restarting, and `--reuse-forecast` completes the remaining stages in ~80 seconds. Checkpoint keys carry a configuration fingerprint, so a change to SKU count, seed, folds or model hyperparameters can never silently reuse stale predictions.

**Requirements:** Python 3.10+. `prophet`, `pmdarima` and `shap` are optional — the system detects what is available and falls back gracefully, reporting which engines ran.

---

## The business problem

In FMCG and modern retail, S&OP failures cost 2–5% of revenue annually through stockouts, excess write-offs and emergency airfreight. Three failure modes recur:

| Failure | Consequence |
|---|---|
| Statistical forecasting ignores market intelligence | The model cannot know about a listing win or an NPD launch |
| Commercial overrides are applied without governance | Optimism enters the plan untracked and unattributable |
| Supply planning ignores capacity constraints | A plan is published that the supply base cannot physically meet |

The monthly S&OP review then becomes a PowerPoint exercise rather than a data-driven decision process. This system addresses all three.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER  (data/raw)                      │
│   11 integrated datasets · consistent keys · schema-validated      │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  MODULE 1 · DEMAND ENGINE            src/planning/demand_engine.py │
│  Holt-Winters │ SARIMA │ Prophet │ XGBoost → inverse-MAPE ensemble │
│  Rolling-origin CV → champion selection → untouched 12wk holdout   │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  MODULE 2 · CONSENSUS                    src/planning/consensus.py │
│  Statistical + Commercial + Constraints → 3 scenarios + audit log  │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  MODULE 3 · SUPPLY PLANNER          src/planning/supply_planner.py │
│  MRP netting · lead-time phasing · capacity · Supply Risk Index    │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  MODULE 4 · OTIF & SPC                 src/planning/kpi_tracker.py │
│  OTIF by 5 dimensions · I-chart · Western Electric · Pareto        │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌───────────────────────────┬────────────────────────────────────────┐
│  MODULE 5 · EXCEL PACK    │  MODULE 6 · DASHBOARD                  │
│  workbook.py — 9 sheets   │  dashboard.py — 5 tabs · static 6 tabs │
└───────────────────────────┴────────────────────────────────────────┘
                    ▲
        src/planning/insights.py — dynamic executive findings
```

### Project structure

```
sop-intelligence-system/
├── src/
│   ├── planning/
│   │   ├── config.py           # every threshold and weight, centralised
│   │   ├── demand_engine.py    # Module 1 — forecasting + ensemble
│   │   ├── consensus.py        # Module 2 — consensus + governance
│   │   ├── supply_planner.py   # Module 3 — MRP + risk scoring
│   │   ├── kpi_tracker.py      # Module 4 — OTIF + SPC
│   │   ├── insights.py         # dynamic insight engine
│   │   ├── workbook.py         # Module 5 — the 9-sheet workbook
│   │   ├── excel_charts.py     # chart + hyperlink primitives
│   │   ├── what_if_model.py    # live Excel formula model (sheet 7)
│   │   ├── saop_pack.py        # CSV export layer + exec pack
│   │   ├── tableau_export.py   # star schema + .hyper writer
│   │   └── pipeline.py         # orchestrator
│   └── shared/
│       ├── data_generator.py   # synthetic supply chain
│       ├── metrics.py          # every KPI, formula-documented
│       ├── validation.py       # temporal + schema validation
│       └── utils.py            # logging, seeding, IO
├── dashboard/
│   ├── dashboard.py            # Module 6 — Plotly Dash, 5 tabs
│   └── static_export.py        # single-file offline HTML, 6 tabs
├── scripts/verify_outputs.py   # artefact verification (chart XML, .hyper)
├── docs/index.html             # the offline dashboard, served by GitHub Pages
├── .github/workflows/tests.yml # CI: pytest + --quick smoke + verification
├── data/raw/                   # 11 generated datasets
├── data/processed/             # 31 CSV exports
├── outputs/                    # workbook · dashboard · tableau/ · manifest
├── tests/                      # 127 pytest tests
├── run_demo.py
├── requirements.txt
├── README.md · CASE_STUDY.md · METHODOLOGY.md
```

---

## Machine learning methodology

### Validation design — the decision that matters most

```
|<------------- training ------------->|<-- CV folds -->|<-- holdout -->|
0                                      67              91             104
```

- **CV folds** drive model selection and ensemble weighting.
- **The final 12 weeks are opened once**, after every decision is frozen.
- Model selection can never see the window accuracy is reported from.

Selecting a champion on the same window you report from is the most common way portfolio forecasting projects overstate results. Here it is structurally impossible, and `assert_no_leakage()` is asserted in the test suite.

### Four models, honestly compared

| Model | Approach | Scope |
|---|---|---|
| Holt-Winters | Damped additive, log space, Duan smearing | All 500 SKUs |
| SARIMA | `auto_arima` AICc search, order cached per SKU | 24-SKU panel |
| Prophet | Indian festival holiday + promo regressors | 24-SKU panel |
| **XGBoost** | Global direct multi-horizon, 46 features | All 500 SKUs |
| Seasonal Naive | Same week last year — **the FVA benchmark** | All 500 SKUs |

SARIMA and Prophet cost 1–4s per fit, so they run on a **revenue-and-category stratified** panel. Sampling the top 24 SKUs would flatter every model — high-volume series are easier to forecast. Panel results are reported as panel results, never extrapolated.

### Why global XGBoost

One model serves all 500 SKUs. Beyond being ~500× cheaper, cross-learning is a real accuracy advantage: a new or low-volume SKU inherits the promotion and festival response estimated across the whole portfolio rather than learning it from thin history.

**Direct multi-horizon** — each row pairs features at origin `t` with the target at `t+h`, with `h` as a feature. Versus recursive forecasting this removes error compounding and an entire class of leakage bugs, since no prediction is fed back as though observed.

### Prediction intervals

```
σ_h = σ₁ · √(1 + λ(h − 1))
```

`σ₁` from CV residuals; **λ is fitted** from the observed long-vs-short horizon variance ratio, not assumed. Coverage is then measured on the untouched holdout and reported as measured.

---

## S&OP methodology

### Consensus governance

1. **The statistical baseline is never overwritten** — it survives as its own column, so a plan miss can be attributed to the model or to the override.
2. **Deviation beyond ±20%** triggers an alert and requires justification; beyond ±45% requires S&OP board sign-off.
3. **Flagged overrides without valid justification are `BLOCKED`.**
4. **Constraints bind last** — commercial ambition is recorded in full, then the capacity ceiling applied. The gap is *constrained-out volume*: revenue the business is choosing not to chase.
5. Every row carries a UTC audit timestamp.

| Scenario | Commercial delta accepted | Risk appetite |
|---|---|---|
| Conservative | 35% | ×0.96 |
| Balanced | 70% | ×1.00 |
| Aggressive | 100% | ×1.05 |

### Fair-share capacity

Supplier capacity is a **shared** resource, allocated pro-rata to demand. Applying full supplier capacity to each SKU independently — the common shortcut — overstates deliverable supply by roughly the number of SKUs per supplier.

---

## Supply planning logic

```
Gross Requirement = Consensus Demand + Safety Stock top-up
Net Requirement   = max(Gross − OnHand − OpenPOs − PlannedReceipts, 0)
Planned Order     = ceil(Net / MOQ) × MOQ
Receipt week      = order week + ⌈lead time⌉
```

**Time-phasing is real.** An order placed inside the lead-time window cannot rescue a stockout in that window — the engine detects this and routes to *expedite* rather than assuming instant supply. Distinguishing recoverable from unrecoverable shortages is the single most useful output for a planner.

**Safety stock** includes lead-time variance and is sized on *forecast error*, not historical demand variation — the buffer exists to absorb error in the plan:

```
SS = Z · √(LT · σ_D² + μ_D² · σ_LT²)
```

Service level is set by ABC class (A 99%, B 95%, C 90%). Holding 99% on a C-item is among the most expensive habits in inventory management.

### Supply Risk Index (0–100, transparent)

```
SRI = 100 × (0.32·stockout_proximity + 0.22·capacity_util + 0.18·cover_shortfall
           + 0.14·lead_time_risk + 0.14·forecast_uncertainty)
```

Green < 35 · Amber 35–65 · Red ≥ 65. Every factor traces to an inspectable number.

---

## OTIF and SPC methodology

```
OTIF = (On Time AND In Full) / eligible orders
```

Flags are **recomputed from source dates and quantities**, never trusted from the ERP field — in practice an ERP's OTIF flag is often computed against a *revised* promise date, quietly laundering lateness out of the metric.

### Why control charts, not thresholds

A process with a true mean of 92% and σ of 2% produces a month at 89% roughly one time in twenty **purely by chance**. Reacting to it is what Deming called *tampering* — it adds variation rather than removing it.

Sigma is estimated from the average moving range (`MR̄ / 1.128`), **not** the sample standard deviation, which is inflated by the very special causes the chart exists to detect. All four Western Electric rules are implemented; Rules 2–4 matter more than Rule 1 because they catch a process that has quietly shifted and is producing points that are each individually unremarkable.

Root causes are ranked by **value at risk**, not frequency — the most common failure is rarely the most expensive.

---

## Dashboards

Two dashboards, both reading the **same published CSVs**, so their numbers cannot diverge.

### A · Offline HTML dashboard — `outputs/sop_dashboard.html`

A **single self-contained file** (~5 MB) containing 17 interactive Plotly charts, KPI cards, sortable/filterable tables and the full insight register. Opens in any browser with no Python, no server and no install. Email it, commit it, or publish it to GitHub Pages.

Six tabs: Overview · Forecast & ML · Consensus Plan · Supply Constraints · OTIF & SPC · Insights. Zoom, hover, legend toggling, table sort and table search all work offline.

This exists because a Dash app requires someone to run it, which makes it useless as a *deliverable*. The static export solves distribution without pretending to be the live tool — server-dependent behaviour (drill-down across 500 individual SKU series) is pre-aggregated rather than faked.

```bash
python dashboard/static_export.py     # rebuild from existing outputs
```

### B · Live Dash application — `http://127.0.0.1:8050`

Five tabs, 26 callbacks, full server-side filtering. Every figure is rendered from a CSV written by the pipeline — **the dashboard cannot invent a KPI**, and a number shown here is provably the same number in the Excel pack because both read the same file.

| Tab | Contents |
|---|---|
| **1 · Forecast vs Actual** | Forecast with 80%/95% bands, SKU/category/channel/horizon filters, accuracy-by-level, model leaderboard, feature importance, SHAP |
| **2 · Consensus Plan** | Waterfall bridge, scenario selector, override distribution vs threshold, governance audit table |
| **3 · Supply Constraints** | Supplier capacity heatmap, risk distribution, inventory trajectory, cover-vs-risk scatter, alert log |
| **4 · OTIF & SPC** | Control chart with special causes marked, WE violations, OTIF by category/channel/supplier, root-cause Pareto |
| **5 · Executive Insights** | Severity-ranked findings with owners and recommended actions |

All tables support native sort, filter and CSV export.

---

## Outputs

### Workbook — `outputs/SOP_Command_Center.xlsx`

**Nine sheets, 25 charts, 1,153 live Excel formulas, 0.15 MB.**

Sheet 1 opens with a **live control panel**: three dropdowns (category, risk band, channel) drive six KPI cells and a top-ten ranked table through `SUMPRODUCT` and `INDEX`/`MATCH`. Nothing in the panel is precomputed — change a filter and Excel re-aggregates and re-ranks. A static report answers the question the analyst thought of; a filtered panel answers the question the reader has.

Verified: at `(All)/(All)/(All)` the panel reconciles exactly to the pipeline manifest — revenue at risk ₹5.70 Cr, 162 Red SKUs, 62 expedites, consensus ₹36.55 Cr. `scripts/verify_outputs.py` re-checks this on every run by summing the staged block and comparing it to `run_manifest.json`.
 Chart-led by design: each sheet leads with two to four charts and carries at most one ranked exception table capped at a screenful. The complete row-level detail lives in the CSV exports, which is where filtering and pivoting actually belong.

| Sheet | Question it answers |
|---|---|
| **1 Dashboard** | Whole-business position on one screen — 8 KPI tiles, 6 charts, top decisions |
| **2 Demand** | Is the forecast good enough to plan on? Accuracy ladder, model competition, forecast interval, category bias |
| **3 Consensus** | What did we agree to, and who changed it? Waterfall, scenarios, override governance |
| **4 Supply & Risk** | What will break, when, and can we still fix it? Inventory trajectory, capacity, expedite list |
| **5 Service & SPC** | Are we delivering, and is the process stable? Control chart, Pareto, supplier scorecard |
| **6 Action Plan** | What must someone do on Monday? Exposure ranked by value, prioritised actions |
| **7 What-If Model** | **97 live Excel formulas** — pick a SKU, edit assumptions, watch safety stock and cost recalculate |
| **8 Guide & Method** | How to read this, key formulas, method, run manifest |
| **9 Master Data** | What data did this run consume? Raw inventory, raw extract, cleaning log, cleaned master (`--full-master-data` writes every row) |

Every sheet states its question in the subtitle, so a reader knows within two seconds whether it is the sheet they want. Consistent furniture throughout: same KPI strip, same two-column chart grid, same colours meaning the same things.

### Tableau data pack (`outputs/tableau/`)

**A native `.hyper` extract plus a full star schema** — for building your own views rather than reading someone else's.

| File | Purpose |
|---|---|
| `SOP_Analytics.hyper` | **17-table native extract, 129,382 rows.** Open directly in Tableau Desktop or Public |
| `data/*.csv` | The same 17 tables as CSV |
| `field_dictionary.csv` | Every field with suggested Dimension/Measure role, type, cardinality, null rate |
| `README.md` | Data model, relationship keys, 10 suggested starting views, calculated fields |

Two ways in, on purpose: **`flat_sku_week`** (55 columns, fully denormalised) for a chart in ninety seconds, and the **star schema** — 4 conformed dimensions, 13 facts at explicit grains — once the flat table stops being enough.

Modelling choices that matter in Tableau specifically:

- **`fact_accuracy_long` is long, not wide** (`Metric` / `Metric_Value`), so one filter drives every accuracy view instead of a worksheet per measure
- **`fact_orders` carries `OTIF_Flag` as an integer** beside the boolean — `AVG([OTIF Flag])*100` gives the OTIF rate in one click
- **`dim_date` spans history and forecast**, separated by `Period_Type`, so actuals and forecast sit on one continuous axis
- Percentages stored already multiplied by 100; dates ISO-8601; booleans real booleans

Verified by re-opening the extract and querying it: `AVG(OTIF_Flag)*100` returns **88.68%** and revenue at risk **₹5.70 Cr**, both matching the pipeline manifest exactly.

### CSV exports (`data/processed/`)

31 datasets — the complete audit trail. The workbook is the decision layer; these are the detail layer. Includes `forecast_uncertainty.csv`, the formal contract to Output 5.

---

## Integration contracts

### Published to Output 5 (Inventory Optimisation)

| Field | Purpose |
|---|---|
| `SKU_ID`, `Supplier_ID`, `Warehouse_ID` | Join keys |
| `Forecast_Units_12W`, `Avg_Weekly_Forecast` | Demand input |
| **`Forecast_Error_Std`** | **σ_D for safety stock** |
| `Historical_Demand_Std`, `Demand_CV` | Variability context |
| `Horizon_Variance_Lambda` | Uncertainty growth |
| `Unit_Cost`, `Unit_Price` | Financial inputs |

Publishing forecast **error** standard deviation rather than historical demand variation is the technically correct input — the buffer absorbs error in the plan, not total demand variation.

### Consumed from Output 1 (Strategic Sourcing) — optional

Supplier IDs, capacity, lead time, MOQ, reliability, sourcing quadrant. Output 2 runs standalone with generated supplier master data; Output 1's intelligence can replace it without code changes.

---

## Engineering quality

| Property | Implementation |
|---|---|
| **Tests** | 127 pytest tests — metrics against hand calculations, MRP balance closure, leakage assertions, governance rules, integration contracts |
| **Reproducibility** | Single seed through one `Generator`; asserted in tests |
| **No hard-coded results** | Every threshold in `config.py`; every KPI computed |
| **Data validation** | Schema, referential integrity, range and business-rule checks before any downstream processing |
| **Leakage prevention** | Rolling-origin splits; `assert_no_leakage()`; feature causality assertion |
| **Graceful degradation** | Every optional dependency has a working fallback |
| **Logging** | Structured, staged, to console and `outputs/logs/pipeline.log` |
| **Type hints** | Throughout public APIs |

Three real bugs were caught during development and fixed: an ambiguous lag definition, a rounding path that could publish a plan 0.05 units above the capacity ceiling, and a checkpoint key that would have let a 60-SKU test cache be loaded as a 500-SKU production forecast.

---

## Key findings from a representative run

*(Exact figures vary with seed; `outputs/run_manifest.json` holds the current run.)*

**Forecast accuracy is reported at every aggregation level, because error cancels as demand aggregates:**

| Level | WAPE | Meets <15% benchmark |
|---|---|---|
| Total Portfolio / 12-Week | 1.07% | ✅ |
| Channel / Month | 4.32% | ✅ |
| Category / Month — *the S&OP sign-off level* | **7.42%** | ✅ |
| Category / Week | 9.60% | ✅ |
| SKU / 12-Week Total | 19.37% | ❌ |
| SKU / Week — *the hardest cut* | 31.25% | ❌ |

**This ladder is the honest answer, and a more defensible one than a single number.** Quoting only the national-monthly figure flatters a bad system; quoting only SKU-week makes a good one look broken. SKU-week WAPE of ~31% is industry-typical for weekly FMCG.

- **Forecast Value Added: +18.2 points** of MAPE over a seasonal-naive benchmark at SKU-week level — the pipeline demonstrably earns its cost.
- **Champion model mix across 500 SKUs:** XGBoost 355, Ensemble 118, Holt-Winters 24, SARIMA 2, Prophet 1. Selection is per SKU on cross-validation performance, and the ensemble does not win by default.
- **Prediction interval coverage:** 80% band achieved 87.4%, 95% band achieved 94.1% — slightly conservative at the 80% level, reported as measured rather than re-tuned.
- **OTIF 88.68% (At Risk)**, with SPC isolating November 2025 as genuine special-cause variation below the control limit — distinct from the routine variation around it.
- **Supply:** 162 Red-risk SKUs, 62 requiring expedite because the shortage falls inside the supplier lead time, 67 capacity breaches, ₹5.70 Cr revenue at risk, ₹1.40 Cr of demand constrained out.
- **Three of seven root causes drive ~68%** of service-failure value, led by supplier delay (32% of failures).

---

## Limitations

1. **Synthetic data** demonstrates methodology, not real-world performance.
2. **SARIMA and Prophet run on a 24-SKU panel**, not the full portfolio.
3. **Holt-Winters uses `seasonal_periods=13`** — 104 weekly observations cannot support a 52-period seasonal fit on training subsets. This is why it underperforms.
4. **Intervals assume symmetric errors**; real demand error is right-skewed. Measured coverage is reported rather than assumed.
5. **Single-echelon MRP** — multi-echelon optimisation is Output 5's scope.
6. **Revenue at risk assumes lost sales**, not deferred demand (upper bound).
7. **Capacity is a weekly unit ceiling** — no changeover or finite scheduling.
8. **13 SPC points** is adequate but at the low end for stable control limits.

---

## Future enhancements

- Hierarchical forecast reconciliation (MinT / OLS top-down–bottom-up)
- Intermittent-demand models (Croston, TSB) for the slow-moving tail
- Multi-echelon inventory optimisation (Output 5 integration)
- Promotion ROI and cannibalisation modelling
- Supplier risk prediction with SHAP explanations (Output 4 integration)
- MLflow experiment tracking and scheduled retraining
- Conformal prediction intervals for distribution-free coverage guarantees
- Docker packaging and CI/CD

---

## Background

I ran localised S&OP and demand forecasting as a Department Manager at Decathlon Sports India — maintaining >90% stock availability, reducing order errors by 35% and cutting inbound logistics delays by 30%. I experienced the S&OP problem operationally: consensus built in spreadsheets, overrides applied without an audit trail, capacity constraints discovered too late to act on.

This project is the analytics system I would have wanted. It encodes that workflow — statistical baseline, governed commercial override, constrained supply plan, Six Sigma service control — as software.

Certifications applied here: **Lean Six Sigma Black Belt** (SPC, control charts, Western Electric rules, Pareto analysis), **Google Cloud Data Analytics**, **SAP Professional Fundamentals** (MRP and PO logic), **Supply Chain Logistics** (Rutgers).

**The synthetic data in this repository is not Decathlon data and does not represent any real organisation.**

---

## Licence

MIT — see `LICENSE`.
