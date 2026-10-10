"""
Synthetic supply-chain data engine.

Generates an internally consistent FMCG/retail dataset: SKU and supplier
masters, 24 months of weekly demand, promotions, price history, Indian festive
calendar, stockout censoring, on-hand inventory, open purchase orders and
5,000 historical delivery orders.

Design principles
-----------------
* **Structure before noise.** Demand is built as an explicit multiplicative
  decomposition -- base x trend x seasonality x festive x promo x price
  elasticity x noise -- so every effect a model later has to recover was
  genuinely put there, and the difficulty of the forecasting problem is a
  deliberate choice rather than an accident.
* **Observed != true.** ``Demand_Units`` is latent true demand; ``Sales_Units``
  is what the ERP would have recorded, censored during stockouts. Models are
  trained on the censored series, exactly as in reality, and accuracy is
  measured against it too.
* **Referential integrity.** Every SKU maps to a supplier, a warehouse and a
  category; every order and PO points at real master data. Output 4 and
  Output 5 of the Command Center can join on these keys without remapping.
* **Determinism.** All randomness flows from one seeded ``np.random.Generator``.

IMPORTANT: this data is simulated for portfolio demonstration. It is not
derived from, and does not represent, any real employer's data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.planning.config import AppConfig, DataConfig, Keys
from src.shared.utils import get_logger, save_csv, set_global_seed, summarise_frame

logger = get_logger("data_generator")


# --------------------------------------------------------------------------- #
# Indian festive calendar
# --------------------------------------------------------------------------- #
# Dates are the observed Gregorian dates of the principal Indian retail demand
# events. Lunar festivals shift year to year, which is exactly why a naive
# "week 44 = Diwali" seasonal index fails and a Prophet holiday regressor
# earns its keep.

FESTIVAL_CALENDAR: Dict[str, List[str]] = {
    "Makar Sankranti / Pongal": ["2024-01-15", "2025-01-14", "2026-01-14"],
    "Republic Day Sale":        ["2024-01-26", "2025-01-26", "2026-01-26"],
    "Holi":                     ["2024-03-25", "2025-03-14", "2026-03-04"],
    "Eid al-Fitr":              ["2024-04-11", "2025-03-31", "2026-03-20"],
    "Independence Day Sale":    ["2024-08-15", "2025-08-15", "2026-08-15"],
    "Raksha Bandhan":           ["2024-08-19", "2025-08-09", "2026-08-28"],
    "Ganesh Chaturthi":         ["2024-09-07", "2025-08-27", "2026-09-14"],
    "Onam":                     ["2024-09-15", "2025-09-05", "2026-08-26"],
    "Navratri":                 ["2024-10-03", "2025-09-22", "2026-10-11"],
    "Dussehra":                 ["2024-10-12", "2025-10-02", "2026-10-20"],
    "Diwali":                   ["2024-11-01", "2025-10-20", "2026-11-08"],
    "Christmas / New Year":     ["2024-12-25", "2025-12-25", "2026-12-25"],
}

# How many weeks before/after the event demand is elevated, and peak intensity.
FESTIVAL_PROFILE: Dict[str, Tuple[int, int, float]] = {
    #  name                       weeks_before, weeks_after, peak_intensity
    "Makar Sankranti / Pongal":   (1, 0, 0.30),
    "Republic Day Sale":          (1, 0, 0.22),
    "Holi":                       (1, 0, 0.38),
    "Eid al-Fitr":                (2, 0, 0.42),
    "Independence Day Sale":      (1, 0, 0.25),
    "Raksha Bandhan":             (1, 0, 0.28),
    "Ganesh Chaturthi":           (1, 1, 0.35),
    "Onam":                       (2, 0, 0.33),
    "Navratri":                   (1, 2, 0.45),
    "Dussehra":                   (1, 0, 0.40),
    "Diwali":                     (3, 1, 1.00),
    "Christmas / New Year":       (2, 1, 0.48),
}


@dataclass
class SupplyChainData:
    """Container for the full generated dataset."""

    sku_master: pd.DataFrame
    supplier_master: pd.DataFrame
    warehouse_master: pd.DataFrame
    sales_history: pd.DataFrame
    promotions: pd.DataFrame
    price_history: pd.DataFrame
    inventory: pd.DataFrame
    purchase_orders: pd.DataFrame
    orders: pd.DataFrame
    festivals: pd.DataFrame
    supplier_capacity: pd.DataFrame

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "sku_master": self.sku_master,
            "supplier_master": self.supplier_master,
            "warehouse_master": self.warehouse_master,
            "sales_history": self.sales_history,
            "promotions": self.promotions,
            "price_history": self.price_history,
            "inventory": self.inventory,
            "purchase_orders": self.purchase_orders,
            "orders": self.orders,
            "festivals": self.festivals,
            "supplier_capacity": self.supplier_capacity,
        }

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([summarise_frame(df, name) for name, df in self.as_dict().items()])


class SupplyChainDataGenerator:
    """Builds the complete synthetic dataset from a seeded generator."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg: DataConfig = config.data
        self.rng: np.random.Generator = set_global_seed(config.seed)

        end = pd.Timestamp(self.cfg.history_end).normalize()
        # Align to Monday-start weeks.
        end = end - pd.Timedelta(days=int(end.dayofweek))
        self.week_starts: pd.DatetimeIndex = pd.date_range(
            end=end, periods=self.cfg.history_weeks, freq="W-MON"
        )
        self.n_weeks = len(self.week_starts)

    # ------------------------------------------------------------------ #
    # Calendar
    # ------------------------------------------------------------------ #
    def build_festival_frame(self) -> pd.DataFrame:
        """
        Expand the festival calendar into a weekly intensity signal.

        Intensity decays linearly across the build-up window and drops faster
        after the event, mirroring observed FMCG sell-in behaviour where the
        trade loads ahead of the festival and de-stocks immediately after.
        """
        rows: List[Dict[str, object]] = []
        # Extend one year beyond history so the forecast horizon has regressors.
        horizon_end = self.week_starts[-1] + pd.Timedelta(weeks=60)

        for name, dates in FESTIVAL_CALENDAR.items():
            before, after, peak = FESTIVAL_PROFILE[name]
            for raw in dates:
                event = pd.Timestamp(raw)
                if event < self.week_starts[0] - pd.Timedelta(weeks=8) or event > horizon_end:
                    continue
                event_week = event - pd.Timedelta(days=int(event.dayofweek))
                for offset in range(-before, after + 1):
                    week = event_week + pd.Timedelta(weeks=offset)
                    if offset < 0:
                        # Ramp up: fullest in the week immediately preceding.
                        intensity = peak * (1.0 - 0.35 * (abs(offset) - 1) / max(before, 1))
                    elif offset == 0:
                        intensity = peak
                    else:
                        intensity = peak * max(0.0, 0.45 - 0.20 * (offset - 1))
                    rows.append({
                        "Festival": name,
                        "Event_Date": event.date().isoformat(),
                        Keys.WEEK: week,
                        "Week_Offset": offset,
                        "Intensity": round(float(max(intensity, 0.0)), 4),
                    })

        frame = pd.DataFrame(rows)
        weekly = (frame.groupby(Keys.WEEK, as_index=False)
                       .agg(Festival_Intensity=("Intensity", "sum"),
                            Festival_Count=("Festival", "nunique"),
                            Festival_Names=("Festival", lambda s: " | ".join(sorted(set(s))))))
        return frame.merge(weekly, on=Keys.WEEK, how="left")

    def _weekly_festival_signal(self, festivals: pd.DataFrame) -> pd.DataFrame:
        weekly = (festivals.groupby(Keys.WEEK)
                  .agg(Festival_Intensity=("Intensity", "sum"),
                       Festival_Names=("Festival", lambda s: " | ".join(sorted(set(s)))))
                  .reset_index())
        weekly["Festival_Intensity"] = weekly["Festival_Intensity"].clip(upper=1.6)
        return weekly

    # ------------------------------------------------------------------ #
    # Master data
    # ------------------------------------------------------------------ #
    def build_supplier_master(self) -> pd.DataFrame:
        """
        Suppliers span three archetypes with genuinely different risk profiles:
        domestic converters (short LT, small MOQ, high reliability), regional
        contract manufacturers, and import sources (long LT, large MOQ,
        volatile). Downstream risk scoring depends on this heterogeneity.
        """
        n = self.cfg.n_suppliers
        rng = self.rng
        archetype = rng.choice(
            ["Domestic", "Regional CM", "Import"], size=n, p=[0.45, 0.35, 0.20]
        )

        lead_time = np.where(
            archetype == "Domestic", rng.uniform(1.0, 2.5, n),
            np.where(archetype == "Regional CM", rng.uniform(2.5, 4.5, n),
                     rng.uniform(5.0, 9.0, n)))
        lt_std = np.where(
            archetype == "Domestic", rng.uniform(0.10, 0.35, n),
            np.where(archetype == "Regional CM", rng.uniform(0.30, 0.75, n),
                     rng.uniform(0.80, 1.80, n)))
        reliability = np.clip(
            np.where(archetype == "Domestic", rng.normal(0.970, 0.020, n),
                     np.where(archetype == "Regional CM", rng.normal(0.940, 0.035, n),
                              rng.normal(0.890, 0.050, n))), 0.70, 0.998)

        countries = np.where(archetype == "Import",
                             rng.choice(["China", "Vietnam", "Indonesia", "UAE", "Thailand"], n),
                             "India")

        df = pd.DataFrame({
            Keys.SUPPLIER: [f"SUP_{i:03d}" for i in range(1, n + 1)],
            "Supplier_Name": [f"{a} Partner {i:03d}" for i, a in enumerate(archetype, start=1)],
            "Archetype": archetype,
            "Country": countries,
            "Lead_Time_Weeks": np.round(lead_time, 2),
            "Lead_Time_Std_Weeks": np.round(lt_std, 2),
            "MOQ": (rng.choice([100, 250, 500, 1000, 2000, 5000], n,
                               p=[0.20, 0.22, 0.22, 0.18, 0.12, 0.06])).astype(int),
            "Reliability_Score": np.round(reliability, 3),
            "Quality_PPM": np.round(rng.gamma(2.0, 900, n)).astype(int),
            "Payment_Terms_Days": rng.choice([30, 45, 60, 90], n, p=[0.35, 0.30, 0.25, 0.10]),
            "Is_Single_Source": rng.random(n) < 0.22,
            # Placeholder for Output 1 hand-off; recomputed by the integration
            # layer when Output 1 supplier intelligence is present.
            "Sourcing_Quadrant": rng.choice(
                ["Strategic", "Leverage", "Bottleneck", "Non-Critical"], n,
                p=[0.18, 0.34, 0.20, 0.28]),
        })
        return df

    def build_warehouse_master(self) -> pd.DataFrame:
        rows = []
        for wid, name, region in self.cfg.warehouses:
            rows.append({
                Keys.WAREHOUSE: wid,
                "Warehouse_Name": name,
                "Region": region,
                "Weekly_Inbound_Capacity_Units": int(self.rng.integers(180_000, 420_000)),
                "Storage_Capacity_Units": int(self.rng.integers(900_000, 2_200_000)),
                "Handling_Cost_Per_Unit": round(float(self.rng.uniform(1.8, 4.2)), 2),
            })
        return pd.DataFrame(rows)

    def build_sku_master(self, suppliers: pd.DataFrame,
                         warehouses: pd.DataFrame) -> pd.DataFrame:
        """
        SKUs get a lognormal base volume (a handful of hero SKUs carry the
        portfolio -- the Pareto shape every real assortment has), a category,
        a primary channel, price/cost economics and a lifecycle status.
        """
        n = self.cfg.n_skus
        rng = self.rng
        categories = rng.choice(list(self.cfg.categories), n)
        channels = rng.choice(list(self.cfg.channels), n,
                              p=list(self.cfg.channel_mix))

        base_volume = np.round(rng.lognormal(mean=5.6, sigma=0.95, size=n)).astype(int)
        base_volume = np.clip(base_volume, 25, 40_000)

        unit_cost = np.round(rng.lognormal(mean=3.4, sigma=0.65, size=n), 2)
        margin = rng.uniform(0.22, 0.58, n)
        unit_price = np.round(unit_cost / (1.0 - margin), 2)

        # Lifecycle: a slice of the portfolio launches mid-history (NPD) and a
        # smaller slice is phased out -- both are the hardest cases to forecast.
        lifecycle = np.array(["Mature"] * n, dtype=object)
        n_npd = int(self.cfg.npd_share * n)
        n_disc = int(self.cfg.discontinued_share * n)
        idx = rng.permutation(n)
        lifecycle[idx[:n_npd]] = "New Product"
        lifecycle[idx[n_npd:n_npd + n_disc]] = "Phase-Out"

        launch_week = np.zeros(n, dtype=int)
        launch_week[idx[:n_npd]] = rng.integers(20, self.n_weeks - 30, n_npd)
        end_week = np.full(n, self.n_weeks, dtype=int)
        end_week[idx[n_npd:n_npd + n_disc]] = rng.integers(
            self.n_weeks - 26, self.n_weeks, n_disc)

        df = pd.DataFrame({
            Keys.SKU: [f"SKU_{i:04d}" for i in range(1, n + 1)],
            "SKU_Name": [f"{c} Item {i:04d}" for i, c in enumerate(categories, start=1)],
            Keys.CATEGORY: categories,
            "Sub_Category": [f"{c[:3].upper()}-{rng.integers(1, 6)}" for c in categories],
            Keys.CHANNEL: channels,
            Keys.SUPPLIER: rng.choice(suppliers[Keys.SUPPLIER].to_numpy(), n),
            Keys.WAREHOUSE: rng.choice(warehouses[Keys.WAREHOUSE].to_numpy(), n),
            "Base_Weekly_Volume": base_volume,
            "Unit_Cost": unit_cost,
            "Unit_Price": unit_price,
            "Price_Elasticity": np.round(
                rng.uniform(*self.cfg.price_elasticity_range, n), 3),
            "Shelf_Life_Days": rng.choice([30, 60, 90, 180, 365, 720], n),
            "Lifecycle_Status": lifecycle,
            "Launch_Week_Index": launch_week,
            "End_Week_Index": end_week,
            "Annual_Trend": np.round(
                [self.cfg.channel_trend[c] * rng.uniform(0.4, 1.6) for c in channels], 4),
            "Noise_CV": np.round(
                [self.cfg.channel_noise_cv[c] * rng.uniform(0.7, 1.4) for c in channels], 4),
        })
        return df

    # ------------------------------------------------------------------ #
    # Demand history
    # ------------------------------------------------------------------ #
    def build_sales_history(self, sku_master: pd.DataFrame,
                            festivals: pd.DataFrame
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Vectorised generation of the SKU x week demand panel.

        Returns (sales_history, promotions, price_history).
        """
        rng = self.rng
        n_sku = len(sku_master)
        n_wk = self.n_weeks

        weekly_festival = self._weekly_festival_signal(festivals)
        festival_map = (weekly_festival.set_index(Keys.WEEK)["Festival_Intensity"]
                        .reindex(self.week_starts).fillna(0.0).to_numpy())
        festival_names = (weekly_festival.set_index(Keys.WEEK)["Festival_Names"]
                          .reindex(self.week_starts).fillna("").to_numpy())

        week_index = np.arange(n_wk)
        iso_week = np.array([d.isocalendar()[1] for d in self.week_starts])
        month = self.week_starts.month.to_numpy()

        base = sku_master["Base_Weekly_Volume"].to_numpy(float)[:, None]
        trend_rate = sku_master["Annual_Trend"].to_numpy(float)[:, None]
        noise_cv = sku_master["Noise_CV"].to_numpy(float)[:, None]
        elasticity = sku_master["Price_Elasticity"].to_numpy(float)[:, None]
        categories = sku_master[Keys.CATEGORY].to_numpy()

        # --- trend: compounding weekly growth ---------------------------- #
        trend = (1.0 + trend_rate / 52.0) ** week_index[None, :]

        # --- annual seasonality ------------------------------------------ #
        amp = np.array([self.cfg.seasonal_amplitude[c] for c in categories])[:, None]
        peak = np.array([self.cfg.seasonal_peak_week[c] for c in categories])[:, None]
        seasonal = 1.0 + amp * np.sin(2 * np.pi * (iso_week[None, :] - peak + 13) / 52.0)

        # secondary half-year harmonic for shape realism
        seasonal += 0.35 * amp * np.sin(4 * np.pi * (iso_week[None, :] - peak) / 52.0)

        # --- festive lift -------------------------------------------------- #
        sensitivity = np.array(
            [self.cfg.festive_sensitivity[c] for c in categories])[:, None]
        # Festive responsiveness is a stable SKU ATTRIBUTE, not weekly noise: a
        # gift-pack SKU responds strongly to Diwali every year. Redrawing it
        # each week would inject irreducible noise no model could ever learn,
        # which would understate what a real forecasting system can achieve.
        sku_festive_response = rng.uniform(0.70, 1.35, size=(n_sku, 1))
        festive = (1.0 + sensitivity * sku_festive_response * festival_map[None, :]
                   * rng.normal(1.0, 0.07, size=(n_sku, n_wk)))

        # --- promotions ---------------------------------------------------- #
        # Promotions run in blocks of 1-3 weeks. The seed rate is deflated so
        # that AFTER the block smearing the realised incidence matches the
        # configured `promo_weeks_share`.
        seed_rate = self.cfg.promo_weeks_share / 1.85
        promo_seed = rng.random((n_sku, n_wk)) < seed_rate
        promo_flag = promo_seed.copy()
        for shift in (1, 2):
            carry = np.zeros_like(promo_flag)
            carry[:, shift:] = promo_seed[:, :-shift] & (
                rng.random((n_sku, n_wk - shift)) < (0.55 if shift == 1 else 0.30))
            promo_flag |= carry

        # Each SKU has its own characteristic promotional lift (a repeatable
        # trade response), varied week to week only by execution noise. This is
        # how promotions actually behave and it is what makes the promo flag a
        # learnable feature rather than a random shock.
        lo, hi = self.cfg.promo_uplift_range
        sku_promo_response = rng.uniform(lo, hi, size=(n_sku, 1))
        promo_uplift = np.where(
            promo_flag,
            sku_promo_response * rng.normal(1.0, 0.09, size=(n_sku, n_wk)),
            1.0)
        promo_uplift = np.maximum(promo_uplift, 0.5)
        promo_depth = np.where(promo_flag, rng.uniform(0.08, 0.35, (n_sku, n_wk)), 0.0)

        # --- price & elasticity ------------------------------------------- #
        list_price = sku_master["Unit_Price"].to_numpy(float)[:, None]
        # Slow inflationary drift plus occasional step changes.
        inflation = (1.0 + rng.uniform(0.02, 0.07, (n_sku, 1)) / 52.0) ** week_index[None, :]
        step_change = np.cumsum(
            (rng.random((n_sku, n_wk)) < 0.012) * rng.uniform(-0.06, 0.09, (n_sku, n_wk)),
            axis=1)
        price = list_price * inflation * (1.0 + step_change) * (1.0 - promo_depth)
        price = np.maximum(price, sku_master["Unit_Cost"].to_numpy(float)[:, None] * 1.02)
        price_index = price / (list_price * inflation * (1.0 + step_change))
        elasticity_effect = np.power(np.clip(price_index, 0.35, 1.6), elasticity)

        # --- lifecycle gating ---------------------------------------------- #
        launch = sku_master["Launch_Week_Index"].to_numpy()[:, None]
        end = sku_master["End_Week_Index"].to_numpy()[:, None]
        weeks_since_launch = week_index[None, :] - launch
        # Bass-like adoption ramp over the first 16 weeks of a new product.
        ramp = np.clip(weeks_since_launch / 16.0, 0.0, 1.0) ** 0.7
        alive = (week_index[None, :] >= launch) & (week_index[None, :] < end)
        # Phase-out decay in the final weeks.
        decay = np.clip((end - week_index[None, :]) / 18.0, 0.0, 1.0)
        decay = np.where(end < n_wk, decay ** 0.8, 1.0)
        lifecycle_factor = np.where(alive, ramp * decay, 0.0)

        # --- multiplicative noise ------------------------------------------ #
        noise = rng.lognormal(mean=-0.5 * noise_cv ** 2,
                              sigma=noise_cv,
                              size=(n_sku, n_wk))

        true_demand = (base * trend * seasonal * festive * promo_uplift
                       * elasticity_effect * lifecycle_factor * noise)
        true_demand = np.maximum(true_demand, 0.0)

        # --- stockout censoring -------------------------------------------- #
        # Stockouts cluster: a SKU that stocked out last week is far more likely
        # to stock out again. This creates the autocorrelated censoring that
        # makes naive forecasting on shipped volume so dangerous.
        stockout = np.zeros((n_sku, n_wk), dtype=bool)
        base_hazard = self.cfg.stockout_week_share
        prob = np.full(n_sku, base_hazard)
        for w in range(n_wk):
            draw = rng.random(n_sku) < prob
            stockout[:, w] = draw & (lifecycle_factor[:, w] > 0)
            # Hazard triples after a stockout (replenishment takes a lead time
            # to recover) then decays geometrically back to the baseline.
            prob = np.where(stockout[:, w],
                            np.minimum(prob * 3.0, 0.35),
                            np.maximum(prob * 0.45, base_hazard))
        severity = rng.uniform(*self.cfg.stockout_severity_range, (n_sku, n_wk))
        fill = np.where(stockout, 1.0 - severity, 1.0)
        observed_sales = true_demand * fill

        demand = np.round(true_demand, 1)
        sales = np.round(observed_sales, 1)

        # --- assemble long frame ------------------------------------------- #
        sku_ids = np.repeat(sku_master[Keys.SKU].to_numpy(), n_wk)
        frame = pd.DataFrame({
            Keys.SKU: sku_ids,
            Keys.WEEK: np.tile(self.week_starts.to_numpy(), n_sku),
            Keys.WEEK_INDEX: np.tile(week_index, n_sku),
            "ISO_Week": np.tile(iso_week, n_sku),
            "Month": np.tile(month, n_sku),
            "Year": np.tile(self.week_starts.year.to_numpy(), n_sku),
            "Demand_Units": demand.ravel(),
            "Sales_Units": sales.ravel(),
            "Price": np.round(price, 2).ravel(),
            "Price_Index": np.round(price_index, 4).ravel(),
            "Is_Promo": promo_flag.ravel(),
            "Promo_Depth": np.round(promo_depth, 3).ravel(),
            "Is_Stockout": stockout.ravel(),
            "Lost_Sales_Units": np.round(demand - sales, 1).ravel(),
            "Festival_Intensity": np.tile(np.round(festival_map, 4), n_sku),
            "Festival_Names": np.tile(festival_names, n_sku),
            "Is_Festival_Week": np.tile(festival_map > 0.05, n_sku),
        })
        frame = frame.merge(
            sku_master[[Keys.SKU, Keys.CATEGORY, Keys.CHANNEL,
                        Keys.SUPPLIER, Keys.WAREHOUSE, "Unit_Cost", "Unit_Price"]],
            on=Keys.SKU, how="left")
        frame["Revenue"] = (frame["Sales_Units"] * frame["Price"]).round(2)

        # Drop pre-launch rows -- a not-yet-launched SKU has no observation,
        # which is different from an observation of zero.
        frame = frame[frame[Keys.WEEK_INDEX] >= np.repeat(launch.ravel(), n_wk)].copy()
        frame = frame.reset_index(drop=True)

        promotions = self._extract_promotions(frame)
        price_history = frame[[Keys.SKU, Keys.WEEK, "Price", "Price_Index",
                               "Promo_Depth"]].copy()

        return frame, promotions, price_history

    @staticmethod
    def _extract_promotions(sales: pd.DataFrame) -> pd.DataFrame:
        """Collapse promo weeks into a promotion event register."""
        promo = sales.loc[sales["Is_Promo"], [Keys.SKU, Keys.WEEK, Keys.WEEK_INDEX,
                                              Keys.CATEGORY, "Promo_Depth",
                                              "Price", "Sales_Units"]].copy()
        if promo.empty:
            return pd.DataFrame(columns=["Promo_ID", Keys.SKU, Keys.WEEK, "Mechanic",
                                         "Discount_Pct", "Planned_Uplift_Pct"])
        promo = promo.sort_values([Keys.SKU, Keys.WEEK_INDEX]).reset_index(drop=True)
        promo["Promo_ID"] = [f"PROMO_{i:06d}" for i in range(1, len(promo) + 1)]
        mechanics = np.array(["Price Off", "BOGO", "Bundle", "Cashback",
                              "Loyalty Multiplier", "Display + Price Off"])
        rng = np.random.default_rng(11)
        promo["Mechanic"] = rng.choice(mechanics, len(promo))
        promo["Discount_Pct"] = (promo["Promo_Depth"] * 100).round(1)
        promo["Planned_Uplift_Pct"] = (promo["Discount_Pct"] * rng.uniform(
            1.4, 3.1, len(promo))).round(1)
        return promo[["Promo_ID", Keys.SKU, Keys.WEEK, Keys.CATEGORY, "Mechanic",
                      "Discount_Pct", "Planned_Uplift_Pct", "Sales_Units"]]

    # ------------------------------------------------------------------ #
    # Inventory, POs, supplier capacity
    # ------------------------------------------------------------------ #
    def build_supplier_capacity(self, sku_master: pd.DataFrame,
                                sales: pd.DataFrame,
                                suppliers: pd.DataFrame) -> pd.DataFrame:
        """
        Weekly supplier capacity, sized as a multiple of that supplier's recent
        demand load.  Roughly a fifth of suppliers are deliberately given a
        tight ratio so the constraint engine has genuine breaches to find --
        capacity that is never binding proves nothing.
        """
        recent = sales[sales[Keys.WEEK_INDEX] >= self.n_weeks - 13]
        load = recent.groupby(Keys.SUPPLIER)["Demand_Units"].mean().rename("Avg_Weekly_Demand")
        load = load.reindex(suppliers[Keys.SUPPLIER]).fillna(load.median() if len(load) else 1000.0)

        n = len(suppliers)
        rng = self.rng
        # Capacity ratio: tight for ~20% of the base, comfortable elsewhere.
        tight = rng.random(n) < 0.20
        # Capacity is sized against DEMAND, but MRP raises ORDERS, which bunch
        # in the first periods and are inflated by MOQ rounding. Real supplier
        # agreements carry headroom for exactly this; without it every run
        # would report a capacity crisis that is an artefact of order lumpiness
        # rather than a genuine supply constraint.
        ratio = np.where(tight, rng.uniform(1.05, 1.45, n), rng.uniform(1.9, 3.6, n))

        # Recent demand is per-SKU-week averaged; scale by SKUs served.
        skus_served = sku_master.groupby(Keys.SUPPLIER).size().reindex(
            suppliers[Keys.SUPPLIER]).fillna(1).to_numpy()

        weekly_capacity = np.maximum(
            np.round(load.to_numpy() * skus_served * ratio), 500).astype(int)

        return pd.DataFrame({
            Keys.SUPPLIER: suppliers[Keys.SUPPLIER].to_numpy(),
            "Weekly_Capacity_Units": weekly_capacity,
            "Capacity_Ratio_Design": np.round(ratio, 2),
            "Capacity_Tier": np.where(tight, "Constrained", "Comfortable"),
            "Committed_Utilisation_Pct": np.round(rng.uniform(35, 80, n), 1),
        })

    def build_inventory(self, sku_master: pd.DataFrame,
                        sales: pd.DataFrame,
                        suppliers: pd.DataFrame) -> pd.DataFrame:
        """
        Opening on-hand position, sized relative to each SKU's supply lead time.

        Cover is drawn as ``lead_time + gamma(...)`` rather than from a flat
        distribution. This is how a functioning business actually holds stock:
        an item on a nine-week import lead time carries far more cover than one
        replenished weekly, because the exposure window is longer. A flat cover
        distribution would make every long-lead SKU look critical on day one
        and would say more about the generator than about the planning logic
        being demonstrated.

        The tail is retained deliberately: a slice of the portfolio is left
        genuinely short and another genuinely overstocked, because an S&OP
        review with nothing to review proves nothing.
        """
        rng = self.rng
        recent = (sales[sales[Keys.WEEK_INDEX] >= self.n_weeks - 8]
                  .groupby(Keys.SKU)["Demand_Units"].mean())
        recent = recent.reindex(sku_master[Keys.SKU]).fillna(0.0)

        lead_time = (suppliers.set_index(Keys.SUPPLIER)["Lead_Time_Weeks"]
                     .reindex(sku_master[Keys.SUPPLIER]).fillna(2.0).to_numpy())

        # Base cover tracks the replenishment cycle; the gamma term adds the
        # spread that makes some SKUs short and others excess.
        cover = lead_time * rng.uniform(0.85, 1.45, len(sku_master)) + rng.gamma(
            shape=2.2, scale=1.35, size=len(sku_master))
        cover = np.clip(cover, 0.3, 20.0)
        on_hand = np.round(recent.to_numpy() * cover)

        in_transit = np.round(
            recent.to_numpy() * rng.uniform(0.0, 2.2, len(sku_master))
            * (rng.random(len(sku_master)) < 0.45))

        return pd.DataFrame({
            Keys.SKU: sku_master[Keys.SKU].to_numpy(),
            Keys.WAREHOUSE: sku_master[Keys.WAREHOUSE].to_numpy(),
            Keys.SUPPLIER: sku_master[Keys.SUPPLIER].to_numpy(),
            Keys.CATEGORY: sku_master[Keys.CATEGORY].to_numpy(),
            "On_Hand_Units": on_hand.astype(int),
            "In_Transit_Units": in_transit.astype(int),
            "Allocated_Units": np.round(on_hand * rng.uniform(0.0, 0.12,
                                                             len(sku_master))).astype(int),
            "Unit_Cost": sku_master["Unit_Cost"].to_numpy(),
            "Snapshot_Week": self.week_starts[-1],
            "Opening_Cover_Weeks": np.round(cover, 2),
            "Lead_Time_Weeks": np.round(lead_time, 2),
        })

    def build_purchase_orders(self, sku_master: pd.DataFrame,
                              suppliers: pd.DataFrame,
                              inventory: pd.DataFrame) -> pd.DataFrame:
        """Open POs already in the pipeline, landing across the planning horizon."""
        rng = self.rng
        supplier_lt = suppliers.set_index(Keys.SUPPLIER)["Lead_Time_Weeks"]
        supplier_moq = suppliers.set_index(Keys.SUPPLIER)["MOQ"]

        has_po = rng.random(len(sku_master)) < 0.78
        sku_with_po = sku_master.loc[has_po].copy()
        n = len(sku_with_po)
        if n == 0:
            return pd.DataFrame(columns=["PO_ID", Keys.SKU, Keys.SUPPLIER, "Order_Qty",
                                         "Expected_Receipt_Week", "Status"])

        base = inventory.set_index(Keys.SKU)["On_Hand_Units"].reindex(
            sku_with_po[Keys.SKU]).fillna(0).to_numpy()
        qty = np.maximum(
            np.round(np.maximum(base, sku_with_po["Base_Weekly_Volume"].to_numpy()) *
                     rng.uniform(0.4, 1.8, n)), 1)
        moq = supplier_moq.reindex(sku_with_po[Keys.SUPPLIER]).fillna(100).to_numpy()
        qty = np.maximum(qty, moq * rng.choice([1, 1, 2], n))

        lt = supplier_lt.reindex(sku_with_po[Keys.SUPPLIER]).fillna(2.0).to_numpy()
        receipt_week = np.clip(np.round(rng.uniform(0.3, 1.4, n) * lt).astype(int),
                               0, self.config.supply.horizon_weeks - 1)

        return pd.DataFrame({
            "PO_ID": [f"PO_{i:06d}" for i in range(1, n + 1)],
            Keys.SKU: sku_with_po[Keys.SKU].to_numpy(),
            Keys.SUPPLIER: sku_with_po[Keys.SUPPLIER].to_numpy(),
            Keys.WAREHOUSE: sku_with_po[Keys.WAREHOUSE].to_numpy(),
            "Order_Qty": qty.astype(int),
            "Order_Date": self.week_starts[-1] - pd.to_timedelta(
                rng.integers(3, 40, n), unit="D"),
            "Expected_Receipt_Week": receipt_week,
            "Expected_Receipt_Date": [
                self.week_starts[-1] + pd.Timedelta(weeks=int(w) + 1) for w in receipt_week],
            "Unit_Cost": sku_with_po["Unit_Cost"].to_numpy(),
            "Status": rng.choice(["Confirmed", "In Transit", "Awaiting Confirmation"],
                                 n, p=[0.55, 0.30, 0.15]),
        })

    # ------------------------------------------------------------------ #
    # Delivery orders (OTIF universe)
    # ------------------------------------------------------------------ #
    def build_orders(self, sku_master: pd.DataFrame,
                     suppliers: pd.DataFrame) -> pd.DataFrame:
        """
        5,000 inbound delivery orders across the trailing 12 months.

        Lateness and short-shipment are driven by each supplier's reliability
        and lead-time variance, with a deliberate step-change in service in the
        festive quarter -- the pattern an SPC chart should detect as a special
        cause rather than as noise.
        """
        rng = self.rng
        n = self.cfg.n_orders
        cfg_kpi = self.config.kpi

        sku_idx = rng.integers(0, len(sku_master), n)
        chosen = sku_master.iloc[sku_idx].reset_index(drop=True)
        sup = suppliers.set_index(Keys.SUPPLIER)

        reliability = sup["Reliability_Score"].reindex(chosen[Keys.SUPPLIER]).to_numpy()
        lt_std = sup["Lead_Time_Std_Weeks"].reindex(chosen[Keys.SUPPLIER]).to_numpy()

        end = self.week_starts[-1]
        start = end - pd.Timedelta(weeks=51)
        promised_offsets = rng.integers(0, 52 * 7, n)
        promised = pd.to_datetime(start) + pd.to_timedelta(promised_offsets, unit="D")

        month_period = promised.to_period("M")
        # Festive quarter stress: Sep-Nov service degrades systematically.
        festive_stress = np.isin(promised.month, [9, 10, 11]).astype(float) * 0.055

        late_prob = np.clip(1.0 - reliability + festive_stress, 0.01, 0.75)
        is_late = rng.random(n) < late_prob
        delay_days = np.where(
            is_late,
            np.maximum(1, np.round(rng.gamma(1.8, 2.0 + lt_std * 3.5, n))).astype(int),
            -rng.integers(0, 3, n))          # early or on-time deliveries

        actual = promised + pd.to_timedelta(delay_days, unit="D")

        ordered_qty = np.maximum(
            np.round(chosen["Base_Weekly_Volume"].to_numpy() * rng.uniform(0.5, 3.0, n)),
            10).astype(int)
        short_prob = np.clip((1.0 - reliability) * 0.60 + festive_stress * 0.45, 0.005, 0.45)
        is_short = rng.random(n) < short_prob
        fill_ratio = np.where(is_short, rng.uniform(0.55, 0.975, n), 1.0)
        delivered_qty = np.round(ordered_qty * fill_ratio).astype(int)

        on_time = delay_days <= cfg_kpi.on_time_grace_days
        in_full = (delivered_qty / np.maximum(ordered_qty, 1)) >= cfg_kpi.in_full_tolerance
        otif = on_time & in_full

        root_cause = self._assign_root_causes(
            rng, on_time, in_full, chosen[Keys.SUPPLIER].to_numpy(), suppliers)

        return pd.DataFrame({
            Keys.ORDER: [f"ORD_{i:06d}" for i in range(1, n + 1)],
            Keys.SUPPLIER: chosen[Keys.SUPPLIER].to_numpy(),
            Keys.SKU: chosen[Keys.SKU].to_numpy(),
            Keys.CATEGORY: chosen[Keys.CATEGORY].to_numpy(),
            Keys.CHANNEL: chosen[Keys.CHANNEL].to_numpy(),
            Keys.WAREHOUSE: chosen[Keys.WAREHOUSE].to_numpy(),
            "Promised_Date": promised,
            "Actual_Delivery_Date": actual,
            "Order_Month": month_period.astype(str),
            "Ordered_Qty": ordered_qty,
            "Delivered_Qty": delivered_qty,
            "Fill_Rate": np.round(delivered_qty / np.maximum(ordered_qty, 1), 4),
            "Delay_Days": delay_days,
            "On_Time": on_time,
            "In_Full": in_full,
            "OTIF": otif,
            "Root_Cause": root_cause,
            "Order_Value": np.round(delivered_qty * chosen["Unit_Cost"].to_numpy(), 2),
        })

    def _assign_root_causes(self, rng: np.random.Generator,
                            on_time: np.ndarray, in_full: np.ndarray,
                            supplier_ids: np.ndarray,
                            suppliers: pd.DataFrame) -> np.ndarray:
        """
        Attribute a root cause to every failed order.

        Causes are not uniform: import suppliers skew to transportation and
        documentation, domestic converters to warehouse and quality. This gives
        the Pareto analysis something real to find.
        """
        causes = np.array(["On Time In Full"] * len(on_time), dtype=object)
        failed = ~(on_time & in_full)

        archetype = suppliers.set_index(Keys.SUPPLIER)["Archetype"].reindex(
            supplier_ids).to_numpy()

        weights = {
            "Domestic": [0.30, 0.22, 0.14, 0.16, 0.08, 0.08, 0.02],
            "Regional CM": [0.34, 0.14, 0.20, 0.14, 0.08, 0.07, 0.03],
            "Import": [0.26, 0.08, 0.32, 0.10, 0.06, 0.06, 0.12],
        }
        options = list(self.config.kpi.root_causes)

        for arch, prob in weights.items():
            mask = failed & (archetype == arch)
            k = int(mask.sum())
            if k:
                causes[mask] = rng.choice(options, k, p=prob)

        remaining = failed & (causes == "On Time In Full")
        if remaining.any():
            causes[remaining] = rng.choice(options, int(remaining.sum()))
        return causes

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def generate(self, persist: bool = True) -> SupplyChainData:
        logger.info("Generating synthetic dataset: %d SKUs x %d weeks, %d suppliers.",
                    self.cfg.n_skus, self.n_weeks, self.cfg.n_suppliers)

        festivals = self.build_festival_frame()
        suppliers = self.build_supplier_master()
        warehouses = self.build_warehouse_master()
        sku_master = self.build_sku_master(suppliers, warehouses)
        sales, promotions, price_history = self.build_sales_history(sku_master, festivals)
        supplier_capacity = self.build_supplier_capacity(sku_master, sales, suppliers)
        suppliers = suppliers.merge(
            supplier_capacity[[Keys.SUPPLIER, "Weekly_Capacity_Units", "Capacity_Tier"]],
            on=Keys.SUPPLIER, how="left")
        inventory = self.build_inventory(sku_master, sales, suppliers)
        purchase_orders = self.build_purchase_orders(sku_master, suppliers, inventory)
        orders = self.build_orders(sku_master, suppliers)

        data = SupplyChainData(
            sku_master=sku_master,
            supplier_master=suppliers,
            warehouse_master=warehouses,
            sales_history=sales,
            promotions=promotions,
            price_history=price_history,
            inventory=inventory,
            purchase_orders=purchase_orders,
            orders=orders,
            festivals=festivals,
            supplier_capacity=supplier_capacity,
        )

        if persist:
            for name, frame in data.as_dict().items():
                save_csv(frame, f"{name}.csv")
            logger.info("Persisted %d datasets to data/raw.", len(data.as_dict()))

        logger.info("Sales panel: %s rows | total demand %.0f units | promo weeks %.1f%% | "
                    "stockout weeks %.1f%%",
                    f"{len(sales):,}", sales["Demand_Units"].sum(),
                    sales["Is_Promo"].mean() * 100, sales["Is_Stockout"].mean() * 100)
        return data


def generate_supply_chain_data(config: AppConfig, persist: bool = True) -> SupplyChainData:
    """Convenience entry point used by the pipeline and the test suite."""
    return SupplyChainDataGenerator(config).generate(persist=persist)
