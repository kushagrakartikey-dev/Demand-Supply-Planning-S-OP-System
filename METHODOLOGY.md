# METHODOLOGY

Technical reference for every calculation in the S&OP Intelligence System.

Each section states the business meaning, the formula, where it is implemented,
and — where it matters — the design alternative that was rejected and why.

> **Data provenance.** All data in this project is synthetic and generated
> programmatically for portfolio demonstration. It does not originate from, and
> does not represent, any employer, client or real organisation.

---

## 1. Demand generation model

Implemented in `src/shared/data_generator.py`.

Weekly demand for SKU *i* in week *t* is built as an explicit multiplicative
decomposition, so every effect a forecasting model must later recover was
genuinely placed in the data:

```
D(i,t) = Base(i)
       × Trend(i,t)
       × Seasonality(cat(i), t)
       × Festive(i,t)
       × Promotion(i,t)
       × PriceElasticity(i,t)
       × Lifecycle(i,t)
       × Noise(i,t)
```

| Component | Form |
|---|---|
| Trend | `(1 + g_i/52)^t`, `g_i` annualised growth drawn per channel |
| Seasonality | `1 + A_c·sin(2π(w − p_c + 13)/52) + 0.35·A_c·sin(4π(w − p_c)/52)` |
| Festive | `1 + S_c · r_i · I(t) · ε`, `I(t)` = festival intensity, `r_i` SKU response |
| Promotion | `u_i · ε` when on promotion, else 1 |
| Price elasticity | `(P(i,t)/P_list(i,t))^η_i`, `η_i ∈ [−2.4, −0.8]` |
| Lifecycle | Bass-style 16-week adoption ramp; phase-out decay |
| Noise | Lognormal, `σ` = channel-specific CV |

**Design note — promotional and festive response are SKU attributes, not weekly
noise.** An earlier version redrew the promotion uplift independently every
week. That is unrealistic (a given SKU on a given mechanic has a repeatable
trade response) and it injects irreducible noise that no model could ever
learn, which would have understated achievable forecast accuracy. Both are now
stable per-SKU parameters with execution noise layered on top.

### Stockout censoring

The generator distinguishes two series, and this distinction drives everything
downstream:

- `Demand_Units` — latent true demand. Never observable in reality.
- `Sales_Units` — what the ERP records, truncated during stockouts.

Models are trained on `Sales_Units` and accuracy is measured against it.
Training on latent demand would inflate every reported metric.

Stockout hazard is autocorrelated: it triples after a stockout and decays
geometrically back to baseline, producing the clustered censoring that makes
naive forecasting on shipped volume dangerous.

### Indian festive calendar

Twelve festivals with observed Gregorian dates for 2024–2026. Lunar festivals
(Diwali, Holi, Eid, Onam) move several weeks year to year, which is exactly why
a fixed week-of-year seasonal index fails and explicit holiday regressors earn
their place. Intensity ramps up across the build-up window and decays faster
afterwards, mirroring trade forward-buy then de-stock.

---

## 2. Forecast accuracy metrics

Implemented in `src/shared/metrics.py`.

| Metric | Formula | Interpretation |
|---|---|---|
| **MAPE** | `100/n · Σ\|A−F\|/\|A\|`, `A ≠ 0` | Scale-free; undefined at zero demand |
| **sMAPE** | `100/n · Σ 2\|A−F\|/(\|A\|+\|F\|)` | Bounded at 200%; defined at zero |
| **WAPE** | `100 · Σ\|A−F\| / Σ\|A\|` | Volume-weighted; aggregates cleanly |
| **MAE** | `1/n · Σ\|A−F\|` | Average error in units |
| **RMSE** | `√(1/n · Σ(A−F)²)` | Penalises large misses |
| **Bias** | `1/n · Σ(F−A)` | Direction of systematic error |
| **Bias %** | `100 · Σ(F−A)/ΣA` | Bias relative to volume |
| **Tracking Signal** | `Σ(F−A) / MAD` | `\|TS\| > 4` ⇒ model has drifted |
| **FVA** | `MAPE(naive) − MAPE(model)` | Value the analytics actually add |

**Why WAPE is the headline metric.** MAPE excludes zero-demand periods and
explodes on low-volume SKUs: a 20-unit item forecast at 60 units contributes
200% error and outvotes a 200,000-unit item forecast to within 2%. WAPE is
volume-weighted, defined at zero, and aggregates consistently up a product
hierarchy. Both are reported; WAPE is the one to steer on.

### Forecast Value Added

The benchmark is **seasonal naive** — repeat the same week last year.

```
FVA = MAPE(seasonal naive) − MAPE(model)
```

A negative FVA means the modelling pipeline is destroying value and should be
retired. The system reports negative FVA rather than clamping it, because that
finding is the entire reason the metric exists.

---

## 3. Validation design

Implemented in `src/shared/validation.py`, orchestrated in `demand_engine.py`.

The series is partitioned into three **disjoint** blocks:

```
|<------------- training -------------->|<-- CV folds -->|<-- holdout -->|
0                                       67              91             104
```

| Block | Purpose | Used for |
|---|---|---|
| CV folds (rolling origin, expanding window) | Model selection, ensemble weighting | Decisions |
| Final 12 weeks (holdout) | Unbiased evaluation | Reporting only |
| Full history | Final refit | Forward forecast |

**This ordering is the single most important design decision in the project.**
Selecting a champion model on the same window you then report accuracy from is
the most common way forecasting projects overstate results. Here it is
structurally impossible: `rolling_origin_splits(..., holdout=12)` carves the
holdout off before any fold is emitted, and `assert_no_leakage()` is asserted in
the test suite.

Random train/test splitting is never used. Shuffling a time series lets the
model interpolate between observations it has already seen either side of,
producing accuracy that cannot be reproduced in production.

---

## 4. Forecasting models

### Model 1 — Holt-Winters (damped additive, log space)

`ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add", seasonal_periods=13)`

Fitted on `log1p(y)`. FMCG demand shocks are proportional, so an additive model
in logs is multiplicative in units and cannot emit a negative forecast.

**Why `seasonal_periods=13`, not 52.** With 104 weekly observations a 52-period
seasonal component consumes half the sample as initial state, and statsmodels
legitimately refuses to fit it on any training subset. Annual seasonality is
instead carried by the Fourier terms and festival regressors in the ML and
Prophet models — an explicit division of labour rather than a silent
degradation.

Back-transform uses a Duan-style smearing correction:

```
E[y] = expm1(ŷ_log) · exp(σ²/2)
```

Without it, `expm1` returns the conditional **median**. Forecasting the median
of a right-skewed demand distribution systematically under-plans and
manufactures stockouts.

### Model 2 — SARIMA

`pmdarima.auto_arima` on `log1p(y)`, `m=13`, stepwise AICc search bounded to
`p,q ≤ 2`, `P,Q ≤ 1`. The discovered order is cached per SKU and re-applied
through the much faster `SARIMAX` path on subsequent folds. Falls back to a
fixed `(1,1,1)(1,0,0,13)` specification if pmdarima is unavailable.

### Model 3 — Prophet with Indian festival regressors

Multiplicative seasonality, `yearly_seasonality=4`, festivals supplied as
holiday regressors plus continuous `festival_intensity` and `promo` extra
regressors.

Two performance decisions, both deliberate:

- **Holiday window is ±1 week, not ±3.** On weekly data Prophet materialises
  one dummy column per holiday-day; a 29-day window across 12 festivals creates
  ~350 columns and a measured 10× slowdown for no accuracy gain. The build-up
  shape is carried instead by the continuous festival regressor.
- **`uncertainty_samples=0`.** Intervals in this system are produced by one
  calibrated method applied uniformly across all models, so per-model
  Monte-Carlo bands would be both inconsistent and expensive.

### Model 4 — Global XGBoost (direct multi-horizon)

One gradient-boosted model serves the entire portfolio.

**Direct multi-horizon formulation.** Each training row pairs features
observable at origin `t` with the target at `t+h`, and `h` is itself a feature.
Versus recursive forecasting this removes error compounding and — more
importantly — removes an entire class of leakage bugs, because no predicted
value is ever fed back in as though it were observed.

**Why global, not per-SKU.** Beyond being ~500× cheaper, cross-learning is a
genuine accuracy advantage: a new or low-volume SKU inherits the promotion and
festival response estimated across the whole portfolio instead of trying to
learn it from its own thin history.

#### Feature set (46 features)

| Family | Features |
|---|---|
| Origin value | `y_at_origin` |
| Lags | `lag_1, 2, 4, 8, 12, 26, 52` (true lags before origin) |
| Rolling mean | windows 4, 8, 13, 26 |
| Rolling std | windows 4, 8, 13, 26 |
| EWMA | spans 4, 13 |
| Momentum | `roll_4/roll_13`, `roll_4/roll_26` |
| Volatility | `cv_13 = roll_std_13 / roll_mean_13` |
| Density | `promo_density_13`, `stockout_density_13` |
| Calendar (target week) | ISO week, month, sin/cos encodings |
| Fourier (target week) | 3 harmonic pairs, period 52.18 |
| Planning (target week) | promo flag, promo depth, price index, festival intensity |
| Static | category code, channel code, log unit price, weeks since launch |
| Horizon | `h` |

**Forward-known regressors are not leakage.** Promotion grid, price plan and
festival calendar are all owned and known a quarter ahead in a real S&OP
process. A live model would have had exactly this information. Only the target
is withheld. Momentum and CV ratios are scale-free by construction so a single
global model can serve a 30-unit and a 30,000-unit SKU.

### Tiered execution

SARIMA and Prophet cost 1–4 seconds per fit, which does not scale to 500 SKUs
on commodity hardware.

| Tier | Models | Coverage |
|---|---|---|
| Mass tier | Holt-Winters + global XGBoost | All 500 SKUs |
| Competition panel | All four models | 24 SKUs, stratified |

The panel is stratified by **category and revenue quintile**. Sampling the top
36 SKUs by revenue would flatter every model, because high-volume series are
substantially easier to forecast. The panel answers "does the expensive model
earn its cost?"; the mass tier delivers a forecast for every SKU. Panel results
are reported as panel results and are never extrapolated to the portfolio.

---

## 5. Ensemble and model selection

### Inverse-MAPE weighting

```
w_i = (1 / MAPE_i) / Σ_j (1 / MAPE_j)
```

Computed from **cross-validation folds only**. Two guards:

- **Floor** (`MAPE_i ← max(MAPE_i, 1.0)`) prevents an implausibly small error
  from monopolising the blend.
- **Minimum weight** (2%): models below it are dropped and the remainder
  renormalised, keeping the ensemble sparse and explainable to a planner.

Seasonal naive is excluded from the blend — it is the benchmark the ensemble is
measured against, not a member of it.

### Champion selection

Per SKU, the candidate with the lowest mean CV MAPE wins. **The ensemble
competes as one more candidate and does not win by default.** Where a single
model is genuinely better, that model is selected and the leaderboard reports
how often that happens.

---

## 6. Prediction intervals

Empirical and self-calibrating.

Per SKU, `σ₁` is the standard deviation of the champion model's CV residuals at
short horizons. Uncertainty grows with horizon:

```
σ_h = σ₁ · √(1 + λ(h − 1))
P10/P90   = F ± 1.2816 · σ_h
P2.5/P97.5 = F ± 1.9600 · σ_h
```

Lower bounds are floored at zero.

**λ is fitted, not assumed.** The observed variance ratio between long-horizon
(h ≥ 9) and short-horizon (h ≤ 4) residuals across the portfolio implies λ,
which is then floored at the configured value:

```
λ_implied = (Var(resid | h≥9) / Var(resid | h≤4) − 1) / (10.5 − 2.5)
```

Coverage is then measured on the untouched holdout and reported honestly. If
the 80% band achieves 88% coverage, that is stated — not quietly re-tuned until
it reads 80%.

---

## 7. Consensus planning

Implemented in `src/planning/consensus.py`.

```
consensus = (statistical + w_s · (commercial − statistical)) · b_s
final     = min(consensus, supply_ceiling)
```

| Scenario | `w_s` (commercial delta accepted) | `b_s` (risk appetite) |
|---|---|---|
| Conservative | 0.35 | 0.96 |
| Balanced | 0.70 | 1.00 |
| Aggressive | 1.00 | 1.05 |

### Governance rules

1. **The statistical baseline is never overwritten.** It is carried as its own
   column through every stage, so when the plan misses you can tell whether the
   model was wrong or the override was.
2. **Deviation** `= 100 · (commercial − statistical) / statistical`.
   Beyond ±20% ⇒ alert + mandatory justification. Beyond ±45% ⇒ S&OP board
   escalation.
3. **Flagged overrides without a valid justification are `BLOCKED`**, not
   silently accepted.
4. **Constraints bind last.** Commercial ambition is recorded in full, then the
   operational ceiling is applied. The gap — *constrained-out volume* — is a
   headline number, because it is revenue the business is choosing not to chase.
5. Every row carries an ISO-8601 UTC audit timestamp.

### Fair-share capacity allocation

Supplier weekly capacity is a **shared** resource, allocated pro-rata to each
SKU's commercial demand:

```
ceiling(i,t) = min( C_supplier(s) · share(i,t), C_warehouse(w) · share(i,t) )
share(i,t)   = demand(i,t) / Σ_{j ∈ supplier s} demand(j,t)
```

Applying full supplier capacity to every SKU independently — the common
shortcut — overstates deliverable supply by roughly the number of SKUs per
supplier.

**Rounding is applied before the ceiling is re-imposed.** Rounding a constrained
quantity up, even by 0.05 units, publishes a plan the supply base cannot meet.

---

## 8. Supply planning (MRP)

Implemented in `src/planning/supply_planner.py`.

### Netting logic

```
Gross Requirement = Consensus Demand + max(SafetyStock − projected position, 0)
Net Requirement   = max(Gross − OnHand − OpenPOs − PlannedReceipts, 0)
Planned Order     = ceil(Net / MOQ) · MOQ
Receipt week      = order week + ceil(lead time)
Closing           = Opening + Receipts − Demand
```

**Available-to-promise, not on-hand.** Allocated stock is committed to customer
orders and is netted off before planning: `available = on_hand − allocated`.

**Time-phasing is real.** An order raised in week *w* lands in week
*w + ⌈lead time⌉*. An order placed inside the lead-time window cannot rescue a
stockout in that window — the engine detects this and routes to expedite rather
than assuming instant supply.

### Safety stock

```
SS = Z · √( LT · σ_D² + μ_D² · σ_LT² )
ROP = μ_D · LT + SS
```

Two decisions that materially change the answer:

- **Lead-time variance is included.** Dropping the second term — the usual
  spreadsheet shortcut — under-buffers precisely the unreliable suppliers that
  need the buffer most.
- **`σ_D` is forecast-error standard deviation, not historical demand
  variation.** The buffer exists to absorb error *in the plan*, so that is what
  it must be sized against.

Service level is set by ABC class, not applied uniformly:

| Class | Service level | Z |
|---|---|---|
| A | 99% | 2.326 |
| B | 95% | 1.645 |
| C | 90% | 1.282 |

Holding 99% service on a C-item is among the most expensive habits in inventory
management.

### Classification

- **ABC** — cumulative 12-week forecast revenue: A ≤ 70%, B ≤ 90%, C remainder.
- **XYZ** — demand CV: X < 0.5, Y ≤ 1.0, Z > 1.0.

### Weeks of cover

```
WoC = Closing Inventory / mean(forward 4-week demand)
```

Capped at 52 so a near-zero-demand SKU does not return infinity and destroy
category averages.

---

## 9. Supply Risk Index

A transparent 0–100 score. Every factor is normalised to [0,1] before weighting,
so no input can dominate through units alone.

```
SRI = 100 × ( 0.32·stockout_proximity
            + 0.22·capacity_utilisation
            + 0.18·cover_shortfall
            + 0.14·lead_time_risk
            + 0.14·forecast_uncertainty )
```

| Factor | Normalisation | Rationale |
|---|---|---|
| Stockout proximity | `(H − first_stockout_week + 1)/H`; ×0.45 if only below-safety-stock | Nearness matters more than depth — a distant shortage is recoverable |
| Capacity utilisation | peak supplier utilisation / 100, capped at 1 | Shared constraint |
| Cover shortfall | `(target − min_cover)/target`, floored at 0 | Distance from policy |
| Lead-time risk | `LT / 8 weeks`, capped at 1 | Long lead time removes the option to react |
| Forecast uncertainty | `(σ_forecast_error / μ_forecast) / 1.5`, capped | Unpredictable SKUs are riskier at equal cover |

| Band | Range |
|---|---|
| Green | < 35 |
| Amber | 35 – 65 |
| Red | ≥ 65 |

Weights are configured in `SupplyConfig.risk_weights` and asserted to sum to 1.0
in the test suite.

---

## 10. OTIF

```
OTIF = (orders delivered On Time AND In Full) / total eligible orders
On Time = actual delivery date ≤ promised date + grace (grace = 0)
In Full  = delivered qty / ordered qty ≥ 0.98
```

The conjunction is deliberate: a line that is early but short, or complete but
late, is a service failure from the customer's point of view.

**Flags are recomputed from source dates and quantities**, never trusted from
the ERP field. In practice an ERP's own OTIF flag is frequently computed against
a *revised* promise date, which quietly launders lateness out of the metric.

| Benchmark | Threshold |
|---|---|
| World Class | ≥ 95% |
| Acceptable | 90 – 95% |
| At Risk | < 90% |

Every dimensional cut carries an order count and a `Statistically_Reliable` flag
(≥ 20 orders). A supplier at 100% on three orders and one at 94% on four hundred
are not comparable, and a scorecard that ranks them side by side misleads.

---

## 11. Statistical process control

```
Centre line = mean(monthly OTIF)
σ̂           = MR̄ / d₂,   d₂ = 1.128  (moving range of n = 2)
UCL / LCL   = CL ± 3σ̂
MR UCL      = 3.267 · MR̄   (D₄ constant)
```

**Why the moving-range estimator, not the sample standard deviation.** The raw
standard deviation is inflated by the very special causes the chart exists to
detect, which widens the limits and hides the signal. `MR̄/d₂` captures only
short-term common-cause variation. This is verified in the test suite against a
series containing a deliberate outlier.

### Western Electric rules

| Rule | Test | Detects |
|---|---|---|
| 1 | Any point beyond 3σ | Shock |
| 2 | 9 consecutive points same side of centre | Sustained shift |
| 3 | 6 consecutive increasing or decreasing | Drift |
| 4 | 2 of 3 consecutive beyond 2σ, same side | Emerging shift |

Rules 2–4 matter more than Rule 1 in practice: they catch a process that has
quietly shifted and is now producing points that are each individually
unremarkable.

**Why SPC rather than a threshold.** A process with a true mean of 92% and σ of
2% will produce a month at 89% roughly one time in twenty purely by chance.
Reacting to it — launching a supplier review, changing a process — is what
Deming called *tampering*: it adds variation rather than removing it. Control
charts answer the only question worth asking: did the process change, or do
processes vary?

---

## 12. Root cause analysis

Failures are ranked by **value at risk**, not frequency, with a cumulative-80%
Pareto cut identifying the vital few. The most frequent failure mode is rarely
the most expensive, and a corrective action plan built on frequency alone
optimises the cheap tail while costly failures continue.

Each cause is mapped to an accountable function and a corrective action. An
insight with no owner is a fact, not an insight.

---

## 13. Financial calculations

```
Working Capital  = closing inventory units × standard unit cost
Revenue at Risk  = shortage units × unit selling price
Constrained Out  = (unconstrained plan − consensus) × unit price
EOQ              = √(2DS/H),  H = holding_rate × unit_cost
Inventory Turns  = annual COGS / average inventory value
```

Revenue at risk assumes **lost** sales rather than deferred demand, making it an
upper bound. This is stated wherever the number is presented.

---

## 14. Integration contracts

### Consumed from Output 1 (Strategic Sourcing) — optional

`Supplier_ID`, supplier capacity, lead time, MOQ, reliability, sourcing quadrant.
Output 2 runs standalone with generated supplier master data; the interface
allows Output 1's supplier intelligence to replace it without code changes.

### Published to Output 5 (Inventory Optimisation)

`forecast_uncertainty.csv` — the formal hand-off:

| Field | Purpose |
|---|---|
| `SKU_ID`, `Supplier_ID`, `Warehouse_ID` | Join keys |
| `Forecast_Units_12W`, `Avg_Weekly_Forecast` | Demand input |
| `Forecast_Error_Std` | **σ_D for safety stock** |
| `Historical_Demand_Std`, `Demand_CV` | Variability context |
| `Horizon_Variance_Lambda` | Uncertainty growth rate |
| `Unit_Cost`, `Unit_Price` | Financial inputs |

Publishing forecast **error** standard deviation rather than historical demand
standard deviation is the technically correct choice, for the reason given in
§8.

---

## 15. Checkpointing and caching

Forecasting dominates runtime -- minutes, against seconds for every downstream
stage. Two levels of caching make the pipeline restartable and make downstream
iteration fast:

| Level | Location | Contents |
|---|---|---|
| Stage checkpoint | `outputs/cache/stages/` | Per-cutoff model predictions (parquet) |
| Forecast result | `outputs/cache/forecast_result.pkl` | The complete fitted result object |

**Checkpoint keys carry a configuration fingerprint** -- a SHA-1 over seed, SKU
count, history length, horizon, CV folds, panel size, model tiers, lag and
rolling-window specifications and every XGBoost hyperparameter.

This is not decoration. A 60-SKU test run and a 500-SKU production run produce
identical `(stage, origin_index)` keys, so without the fingerprint the test
cache would be silently loaded as though it were the production forecast -- a
wrong answer delivered confidently, which is the worst possible failure mode for
a cache. Any configuration change produces a different key and forces a refit.

---

## 16. Reproducibility

A single seed (`RANDOM_SEED = 20260215`) flows through one
`np.random.Generator`. `set_global_seed()` also seeds `random` and
`PYTHONHASHSEED`. Model random states derive from the same seed. Two runs on the
same machine produce identical numbers; this is asserted in the test suite.

---

## 17. Known limitations

1. **Synthetic data.** Results demonstrate methodology, not real-world
   performance. Real demand contains structural breaks, competitor actions and
   data-quality problems no generator reproduces.
2. **SARIMA and Prophet run on a 36-SKU panel**, not the full portfolio, for
   runtime reasons. Panel results are not extrapolated.
3. **`seasonal_periods=13` in Holt-Winters** limits its annual seasonality
   capture; this is why it underperforms in the leaderboard.
4. **Intervals assume symmetric errors.** Real demand error is right-skewed;
   the empirical calibration partly compensates, and measured coverage is
   reported rather than assumed.
5. **Single-echelon MRP.** No multi-echelon inventory optimisation or
   distribution requirements planning — that is Output 5's scope.
6. **Revenue at risk assumes lost sales**, not deferred demand.
7. **Capacity is modelled as a weekly unit ceiling**, without changeover times,
   sequencing, or finite-scheduling detail.
8. **No lot expiry or shelf-life netting**, despite shelf life being generated.
9. **OTIF is a bounded percentage**, so a computed 3-sigma UCL can exceed 100%.
   The raw sigma-based limits are retained in the `*_3Sigma` columns; the
   charted `UCL`/`LCL` are clipped to the feasible range and flagged with
   `UCL_Truncated_At_Bound`.
10. **12-month order history** gives 13 SPC points — adequate but at the low end
   for stable control limits; the engine warns when below 8.
11. **Forward promotion and price plans are simulated deterministically.** In
    production these come from trade marketing and revenue management systems.
