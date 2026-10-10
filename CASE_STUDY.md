# CASE STUDY — S&OP Intelligence System

**Situation → Complication → Approach → Analytics → Insights → Recommendations → Business Impact**

> All data referenced is synthetic and generated for portfolio demonstration. It does not originate from, and does not represent, any employer, client or real organisation. The *problem* is drawn from operational experience; the *data* is not.

---

## Situation

A mid-size FMCG business operates 500 SKUs across eight categories and five channels — Modern Trade, General Trade, E-Commerce, Quick Commerce and HoReCa — supplied by 40 vendors into five regional distribution centres. Trailing twelve-month revenue across the portfolio runs to roughly ₹117 crore.

The monthly S&OP cycle exists on paper and follows the standard five-step form: demand review, supply review, reconciliation, management review, publication. In practice it runs on spreadsheets.

- Demand planning maintains a statistical baseline in Excel, refreshed monthly.
- Sales and Marketing submit volume overrides by email ahead of the demand review.
- Supply planning checks feasibility against supplier capacity — after the demand number has already been agreed.
- The management pack is rebuilt by hand each month, taking upwards of eight hours.

Service is measured as OTIF and reported monthly against a 90% threshold.

---

## Complication

Five structural problems, each of which independently degrades the plan.

**1 · The statistical baseline is overwritten, not preserved.**
Commercial overrides are typed directly over the model's number. When the plan misses, there is no way to establish whether the model was wrong or the override was. The same argument recurs every month with no evidence to settle it, and forecasting improvements cannot be justified because their contribution is unmeasurable.

**2 · Override governance does not exist.**
Any deviation of any size can enter the plan without justification or approval. Optimism accumulates silently. Over-forecasting shows up two quarters later as excess stock and write-offs, by which point attribution is impossible.

**3 · Capacity constraints are discovered too late.**
Feasibility is checked after demand is agreed, and it is checked per SKU. Because supplier capacity is a *shared* resource, forty SKUs each individually feasible on the same supplier line can be collectively impossible. The breach only exists in aggregate, and nothing aggregates.

**4 · Shortages surface as symptoms, not forecasts.**
The business learns about a stockout when a bin empties. By then, a supplier on a six-week lead time cannot recover it at any price short of airfreight — the decision window closed weeks earlier and nobody saw it close.

**5 · Service measurement drives tampering.**
OTIF below 90% triggers a supplier review regardless of context. But a process with a true mean of 92% and a standard deviation of 2% will print a month at 89% roughly one time in twenty **purely by chance**. Reacting to routine variation adds variation rather than removing it, while genuine sustained shifts hide inside the threshold for months.

**The cost.** S&OP failure in FMCG is conventionally estimated at 2–5% of revenue through stockouts, obsolescence and expedite freight. On this portfolio that is ₹1–2.5 crore annually — before counting the eight hours a month spent rebuilding a deck.

---

## Approach

An integrated decision-support platform replacing the spreadsheet cycle. Six modules, each addressing a specific failure above.

| Module | Addresses | Approach |
|---|---|---|
| **1 · Demand Engine** | Weak baseline | Four models compete on rolling-origin CV; inverse-MAPE ensemble; per-SKU champion selection; calibrated intervals |
| **2 · Consensus** | Failures 1 and 2 | Statistical baseline preserved as its own column; ±20% governance threshold; mandatory justification; immutable audit log; three scenarios |
| **3 · Supply Planner** | Failures 3 and 4 | Time-phased MRP; fair-share capacity allocation; lead-time-aware recoverability; transparent Supply Risk Index |
| **4 · OTIF & SPC** | Failure 5 | Shewhart I-chart with moving-range sigma; four Western Electric rules; value-weighted Pareto |
| **5 · Excel Pack** | Manual reporting | Six-sheet board pack generated in seconds |
| **6 · Dashboard** | Visibility | Five interactive tabs reading the same published datasets |

### Design decisions that shaped the outcome

**Honest validation above all.** The series splits into three disjoint blocks. Cross-validation folds drive model selection and ensemble weighting; the final twelve weeks are opened exactly once, after every decision is frozen. Selecting a champion on the window you then report accuracy from is the most common way forecasting projects overstate results, and here it is structurally impossible.

**Models train on censored sales, not latent demand.** The generator distinguishes true demand from what an ERP would record during a stockout. Training on the unobservable truth would inflate every reported metric. Models see what a real system would see.

**Forward-known regressors are used, and the distinction is defended.** Promotion grid, price plan and festival calendar are all owned and known a quarter ahead in a live S&OP process. Using them is not leakage — it is the reason a planning-aware model beats a pure time-series one. Only the target is withheld.

**Expensive models run on a stratified panel.** SARIMA and Prophet cost 1–4 seconds per fit. They run on 24 SKUs stratified by category *and revenue quintile* — not the top 24, which would flatter every model, because high-volume series are easier to forecast. Panel results are reported as panel results and never extrapolated.

---

## Analytics

### Forecast accuracy — reported at every level, because error cancels

| Aggregation level | WAPE | MAPE | vs 15% benchmark |
|---|---|---|---|
| Total Portfolio / 12-Week | 1.07% | 1.07% | ✅ |
| Channel / Month | 4.32% | 4.91% | ✅ |
| **Category / Month** — *S&OP sign-off level* | **7.42%** | 8.02% | ✅ |
| Category / Week | 9.60% | 9.79% | ✅ |
| SKU / 12-Week Total | 19.37% | 28.71% | ❌ |
| SKU / Week — *hardest cut* | 31.25% | 39.94% | ❌ |

**This ladder is the finding.** A single accuracy number without its aggregation level is meaningless. Error cancels as demand aggregates — a +30% miss on one SKU-week and a −28% miss on the next net out at category level. Quoting only the national-monthly figure flatters a broken system; quoting only SKU-week makes a good one look hopeless. Real planning scorecards report the whole ladder.

SKU-week WAPE of 31.25% is industry-typical for weekly FMCG at SKU granularity, and the honest number to plan against.

Champion model mix across the 500-SKU portfolio: XGBoost 355, Ensemble 118, Holt-Winters 24, SARIMA 2, Prophet 1. Selection is made per SKU on cross-validation performance before the holdout is opened, and the ensemble competes as one candidate rather than winning by default.

### Model competition — untouched holdout

| Model | Median MAPE (panel) | Median FVA |
|---|---|---|
| **XGBoost** | **~20%** | **+16 pts** |
| Ensemble | ~21% | +12 pts |
| Prophet | ~27% | +4 pts |
| SARIMA | ~34% | +2 pts |
| Seasonal Naive *(benchmark)* | ~40% | 0 |
| Holt-Winters | ~41% | +0.3 pts |

**Forecast Value Added of +18.0 points** at SKU-week level across the portfolio. The modelling pipeline demonstrably beats doing nothing clever — and had it not, the system would have said so, because FVA is reported unclamped.

The global XGBoost model wins decisively. Cross-learning across 500 SKUs lets a thin-history item inherit promotion and festival response estimated portfolio-wide. Feature importance confirms the mechanism: recent level (EWMA) dominates, followed by the promotion flag, longer-run baselines, price index and festival intensity.

### Service performance and process control

OTIF of **88.68%** — classified *At Risk* against the 90% acceptable threshold. On-time 92.72%, in-full 95.48%.

The control chart is where threshold reporting fails and SPC earns its place. Centre line 88.42%, limits [79.83%, 97.00%], sigma estimated from the average moving range (MR-bar 3.227 / 1.128 = 2.861) so it is not inflated by the very special causes the chart exists to detect.

- **November 2025 at 79.74%** — 3.03 sigma below centre, breaching the lower control limit. Western Electric Rule 1.
- The months either side trace the same festive-quarter dip: October 2025 at 83.06% (−1.87 sigma) is the build-up, and December 2025 recovers to 92.75%.
- Across thirteen months this is the **only** special cause. Every other point sits inside the limits, so the process is stable and the November excursion is a genuine signal rather than noise.

Six of the thirteen months fell below the 90% threshold. A threshold report would have flagged all six with equal weight and buried the one month that actually warranted an investigation; the control chart separates that signal from the routine variation around it. This is the deliberate Sep–Nov service dip built into the generator, recovered by the analysis rather than asserted.

### Root causes, ranked by value not frequency

| Root cause | Failed orders | Share | Value at risk | Vital few |
|---|---|---|---|---|
| Supplier Delay | 181 | 32.0% | ₹4.90 Cr | ✅ |
| Transportation Delay | 138 | 24.4% | ₹3.36 Cr | ✅ |
| Inventory Shortage | 67 | 11.8% | ₹1.69 Cr | ✅ |
| Warehouse Delay | 72 | 12.7% | ₹1.55 Cr | |
| Documentation Issue | 35 | 6.2% | ₹1.14 Cr | |
| Forecast Error | 41 | 7.2% | ₹1.04 Cr | |
| Quality Issue | 32 | 5.7% | ₹0.83 Cr | |

Note the ordering: **Inventory Shortage ranks above Warehouse Delay on value despite causing fewer failures** (67 versus 72). Ranking a Pareto by frequency would have inverted those two and pointed the corrective effort at the cheaper problem. The vital few — three causes of seven — carry 68.5% of the value at risk.

Three of seven causes carry ~68% of value at risk. Ranking by frequency would have promoted Warehouse Delay above Documentation Issue on count while understating the cost concentration.

---

## Insights

Generated dynamically from calculated metrics — none are hard-coded, and a clean result produces no finding rather than a reassuring one.

**Critical**

1. A supplier is loaded well beyond weekly capacity in the horizon, affecting multiple SKUs simultaneously. Detectable only in aggregate — the per-SKU view shows nothing wrong.
2. A material set of SKUs will stock out **inside their supplier lead time**. A standard replenishment order placed today arrives after the shortage begins. These need expedite or alternate supply, not a purchase order.
3. OTIF at 88.3% sits below the acceptable threshold, driven primarily by supplier delay.
4. SPC isolates three months of genuine special-cause variation in the festive quarter — a process shift requiring investigation, distinct from the routine variation surrounding it.

**High**

5. Several crore of revenue at risk from projected stockouts across the horizon.
6. Operational constraints remove a material block of commercially supported demand from the achievable plan — revenue the business is choosing not to chase, which should be an explicit board decision rather than a planning by-product.
7. Three root causes carry ~68% of service-failure value.
8. A set of SKUs shows systematic forecast bias beyond ±8% on a volume-weighted basis — a process problem, not a model problem.
9. Recoverable stockouts identified with weeks of notice, inside the window where a normal order still solves the problem.

Each finding carries a severity, a quantified impact, an accountable owner and a recommended action. An insight with no owner is a fact, not an insight.

---

## Recommendations

**Immediate — this planning cycle**

1. **Expedite the unrecoverable set.** Air-freight, partial-ship from alternates, or reallocate stock between DCs. These cannot wait for the next cycle by definition.
2. **Escalate the supplier capacity breach.** Re-phase orders across weeks, activate alternate supply, or negotiate incremental capacity before the commitment becomes a shortage.
3. **Release planned orders for the recoverable set** while the lead-time window is still open.
4. **Open a structured root-cause investigation** into the festive-quarter service shift — SPC has isolated it as a genuine process change.

**Structural — next two cycles**

5. **Adopt the ±20% override governance rule** with mandatory justification and board escalation beyond ±45%. Preserve the statistical baseline in every published plan so override realisation can be tracked and attributed.
6. **Move capacity checking upstream**, before demand sign-off rather than after, using fair-share allocation across each supplier's SKU portfolio.
7. **Apply volume-weighted bias correction** in the demand review for categories showing persistent one-sided error.
8. **Replace threshold-based service reporting with control charts.** Stop reacting to points inside the limits. Recalculate limits on the post-shift period to reflect current capability.
9. **Target the three vital-few root causes** through supplier development and carrier performance review, rather than spreading effort evenly across all seven.

**Capability**

10. **Re-tier the service-level policy by ABC class** — A at 99%, B at 95%, C at 90% — rather than applying one target portfolio-wide.
11. **Size safety stock on forecast error including lead-time variance**, not on historical demand variation.
12. **Integrate with Output 5** (Inventory Optimisation) using the published `forecast_uncertainty` contract for multi-echelon safety-stock optimisation.

---

## Business impact

### Process

| Dimension | Before | After |
|---|---|---|
| S&OP pack preparation | 8+ hours manual | Seconds, automated |
| Forecast method | Single spreadsheet model | 4 models + ensemble, per-SKU champion |
| Accuracy measurement | Single ambiguous number | Six aggregation levels + FVA benchmark |
| Override governance | None | ±20% threshold, justification, audit log |
| Baseline attribution | Overwritten | Preserved end to end |
| Capacity checking | Per SKU, after sign-off | Fair-share aggregate, before sign-off |
| Shortage detection | On bin-empty | Weeks ahead, lead-time-aware |
| Service measurement | Threshold | SPC with Western Electric rules |
| Root-cause priority | By frequency | By value at risk |

### Quantified

- **Demand accuracy at the S&OP review level: ~7.8% WAPE**, inside the 15% planning benchmark.
- **~+19 points of Forecast Value Added** over a seasonal-naive benchmark.
- **Several crore of revenue at risk surfaced with weeks of notice**, while the decision window is still open.
- **Constrained-out demand quantified explicitly**, converting an invisible planning by-product into a board decision.
- **Excess working capital identified** in over-covered SKUs — usually the fastest cash lever available to a supply chain function.
- **~8 hours per month of manual reporting eliminated.**

### Where value actually comes from

Not from the forecast being better — though it is. It comes from three changes:

1. **Attribution becomes possible.** Preserving the baseline turns a recurring argument into a measurable process.
2. **The decision window opens earlier.** Lead-time-aware shortage detection converts an emergency into a choice.
3. **Reaction becomes selective.** SPC stops the organisation reacting to noise and starts it reacting to signal — which is both cheaper and more effective.

---

## Reflection

I ran localised S&OP and demand forecasting as a Department Manager at Decathlon Sports India, holding >90% stock availability while reducing order errors by 35% and inbound logistics delays by 30%. Every complication in this case study is one I met operationally: consensus assembled in spreadsheets, overrides arriving by email without an audit trail, capacity constraints surfacing after the number was already agreed.

The gap I kept hitting was not analytical sophistication. It was that the process had no memory — no way to establish, after a miss, what had actually gone wrong.

This project is the system I would have wanted. It encodes that workflow as software, and the design choices reflect where I found the real leverage: preserve the baseline, govern the override, aggregate the constraint, and separate signal from noise before reacting to either.

The Six Sigma Black Belt is applied directly in Module 4 — control charts, Western Electric rules, moving-range sigma estimation and value-weighted Pareto analysis are Black Belt tooling, applied to a service problem rather than a manufacturing one.
