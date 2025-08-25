# Hotel Revenue Simulator — Streamlit App (Customized for Ramada Abu Dhabi Corniche)
# -----------------------------------------------------------------------------
# This version preloads your hotel inventory (235 rooms) and key Abu Dhabi events
# with demand uplifts and temporary elasticity dampening during compression.
#
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# How to deploy: push app.py + requirements.txt to a public GitHub repo and
# deploy with Streamlit Community Cloud (choose app.py as Main file path).
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="Hotel Revenue Simulator — Ramada AD Corniche", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")
pd.options.display.float_format = lambda x: f"{x:,.2f}"

# -------------------------
# Data structures & helpers
# -------------------------
@dataclass
class RoomType:
    name: str
    rooms: int
    base_rate: float
    min_rate: float
    max_rate: float

@dataclass
class Segment:
    name: str
    mix: float           # share of demand (sums ~1.0 across segments)
    elasticity: float    # negative; e.g., -0.5
    commission: float    # fraction; e.g., 0.18 → 18%

# -------------------------
# Default data — customized for your hotel
# -------------------------
# Total rooms = 235
DEFAULT_ROOMS = pd.DataFrame([
    {"room_type": "Deluxe City View",           "rooms": 61, "base_rate": 250.0, "min_rate": 180.0, "max_rate": 380.0},
    {"room_type": "Deluxe Twin City View",      "rooms": 25, "base_rate": 250.0, "min_rate": 180.0, "max_rate": 380.0},
    {"room_type": "Deluxe Double Sea View",     "rooms": 38, "base_rate": 300.0, "min_rate": 210.0, "max_rate": 480.0},
    {"room_type": "Deluxe Twin Sea View",       "rooms": 26, "base_rate": 300.0, "min_rate": 210.0, "max_rate": 480.0},
    {"room_type": "Executive City View",        "rooms": 25, "base_rate": 380.0, "min_rate": 280.0, "max_rate": 650.0},
    {"room_type": "Executive Sea View",         "rooms": 26, "base_rate": 420.0, "min_rate": 300.0, "max_rate": 700.0},
    {"room_type": "People of Determination",    "rooms":  2, "base_rate": 250.0, "min_rate": 180.0, "max_rate": 380.0},
    {"room_type": "Ramada Suite",               "rooms": 32, "base_rate": 650.0, "min_rate": 450.0, "max_rate": 1200.0},
])

# You can edit these in the UI, but here are reasonable starting values
DEFAULT_SEGS = pd.DataFrame([
    {"segment": "OTA",                "mix": 0.2837, "elasticity": -0.80, "commission": 0.18},
    {"segment": "Walk-In",            "mix": 0.0386, "elasticity": -0.30, "commission": 0.00},
    {"segment": "Direct",             "mix": 0.0085, "elasticity": -0.50, "commission": 0.00},
    {"segment": "Discounted Retail-H", "mix": 0.0117, "elasticity": -0.90, "commission": 0.00},
    {"segment": "Corporate",          "mix": 0.4528, "elasticity": -0.20, "commission": 0.05},
    {"segment": "Tour Series",        "mix": 0.0764, "elasticity": -0.50, "commission": 0.10},
    {"segment": "Wholesaler",         "mix": 0.1271, "elasticity": -0.60, "commission": 0.18},
])

DEFAULT_WEEKDAY = pd.DataFrame([
    {"weekday": "Mon", "factor": 1.05},
    {"weekday": "Tue", "factor": 1.05},
    {"weekday": "Wed", "factor": 1.03},
    {"weekday": "Thu", "factor": 1.00},
    {"weekday": "Fri", "factor": 0.98},
    {"weekday": "Sat", "factor": 0.95},
    {"weekday": "Sun", "factor": 0.98},
])

# Pre-populated Abu Dhabi events (2025–2026) with demand uplift and elasticity dampening
# uplift_factor: multiplier to unconstrained demand (1.60 = +60%)
# elasticity_factor: multiply segment elasticities by this (<1 makes price less sensitive during compression)
PREPOP_EVENTS = pd.DataFrame([
    {"name": "ADIHEX 2025", "start_date": pd.to_datetime("2025-08-30"), "end_date": pd.to_datetime("2025-09-07"), "uplift_factor": 1.20, "elasticity_factor": 0.75},
    {"name": "UFC 321 (Showdown Week)", "start_date": pd.to_datetime("2025-10-24"), "end_date": pd.to_datetime("2025-10-26"), "uplift_factor": 1.35, "elasticity_factor": 0.50},
    {"name": "ADIPEC 2025", "start_date": pd.to_datetime("2025-11-03"), "end_date": pd.to_datetime("2025-11-06"), "uplift_factor": 1.60, "elasticity_factor": 0.40},
    {"name": "Abu Dhabi Art 2025", "start_date": pd.to_datetime("2025-11-20"), "end_date": pd.to_datetime("2025-11-24"), "uplift_factor": 1.15, "elasticity_factor": 0.80},
    {"name": "ADIBS 2025 (Boat Show)", "start_date": pd.to_datetime("2025-11-21"), "end_date": pd.to_datetime("2025-11-24"), "uplift_factor": 1.20, "elasticity_factor": 0.75},
    {"name": "Global Media Congress 2025", "start_date": pd.to_datetime("2025-11-26"), "end_date": pd.to_datetime("2025-11-28"), "uplift_factor": 1.25, "elasticity_factor": 0.70},
    {"name": "Mother of the Nation Festival 2025", "start_date": pd.to_datetime("2025-11-22"), "end_date": pd.to_datetime("2025-12-08"), "uplift_factor": 1.10, "elasticity_factor": 0.90},
    {"name": "F1 Etihad Abu Dhabi Grand Prix 2025", "start_date": pd.to_datetime("2025-12-04"), "end_date": pd.to_datetime("2025-12-07"), "uplift_factor": 1.70, "elasticity_factor": 0.35},
    {"name": "ADNOC Abu Dhabi Marathon 2025", "start_date": pd.to_datetime("2025-12-12"), "end_date": pd.to_datetime("2025-12-14"), "uplift_factor": 1.10, "elasticity_factor": 0.85},
    {"name": "World Future Energy Summit 2026", "start_date": pd.to_datetime("2026-01-13"), "end_date": pd.to_datetime("2026-01-15"), "uplift_factor": 1.35, "elasticity_factor": 0.60},
    {"name": "Abu Dhabi International Book Fair 2026", "start_date": pd.to_datetime("2026-04-26"), "end_date": pd.to_datetime("2026-05-05"), "uplift_factor": 1.20, "elasticity_factor": 0.75},
    {"name": "Middle East Film & Comic Con 2026", "start_date": pd.to_datetime("2026-04-10"), "end_date": pd.to_datetime("2026-04-12"), "uplift_factor": 1.15, "elasticity_factor": 0.80},
    {"name": "ADIHEX 2026", "start_date": pd.to_datetime("2026-08-28"), "end_date": pd.to_datetime("2026-09-06"), "uplift_factor": 1.20, "elasticity_factor": 0.75},
])

# Rules for rate recommendation
LOW_OCC_THRESHOLD  = 0.65
HIGH_OCC_THRESHOLD = 0.85
MAX_DOWN_ADJ = 0.12  # max -12%
MAX_UP_ADJ   = 0.15  # max +15%

# Overbooking & cancellations
DEFAULT_OVERBOOK = 5
DEFAULT_CANCELS_NOSHOW = 0.03

# -------------------------
# Utility functions
# -------------------------

def normalize_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "mix" in df.columns and df["mix"].sum() > 0:
        df = df.copy()
        df["mix"] = df["mix"] / df["mix"].sum()
    return df


def generate_synthetic_history(n_days: int, today_occ: float, today_adr: float) -> tuple[list[float], list[float]]:
    occ_hist, adr_hist = [], []
    for i in range(n_days):
        drift = (i - n_days/2) / (n_days*3)
        occ = np.clip(today_occ + np.random.uniform(-0.06, 0.06) + drift, 0.45, 0.92)
        adr = max(80.0, today_adr * np.random.uniform(0.94, 1.06))
        occ_hist.append(float(occ))
        adr_hist.append(float(adr))
    return occ_hist, adr_hist


def moving_average_forecast(series: list[float], horizon: int, window: int = 7) -> list[float]:
    s = list(series)
    out = []
    for _ in range(horizon):
        lookback = s[-window:] if len(s) >= window else s
        ma = sum(lookback) / len(lookback)
        out.append(ma)
        s.append(ma)
    return out


def weekday_index_map() -> dict[int, str]:
    return {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def expand_event_ranges(events_df: pd.DataFrame) -> tuple[dict[date, float], dict[date, float]]:
    """Create date→uplift and date→elasticity_factor maps from ranged events.
    If multiple events overlap, we apply the strongest compression: max uplift, min elasticity_factor.
    """
    uplift_map: Dict[date, float] = {}
    elast_map: Dict[date, float] = {}
    for _, r in events_df.iterrows():
        try:
            sd = pd.to_datetime(r["start_date"]).date() if pd.notnull(r["start_date"]) else None
            ed = pd.to_datetime(r["end_date"]).date() if pd.notnull(r["end_date"]) else sd
            if sd is None:
                continue
            factor = float(r.get("uplift_factor", 1.0))
            ef     = float(r.get("elasticity_factor", 1.0))
            d = sd
            while d <= ed:
                uplift_map[d] = max(uplift_map.get(d, 1.0), factor)
                elast_map[d]  = min(elast_map.get(d, 1.0), ef)
                d += timedelta(days=1)
        except Exception:
            continue
    return uplift_map, elast_map


def apply_weekday_and_events(base_occ: list[float], start: date, weekday_df: pd.DataFrame, events_df: pd.DataFrame) -> tuple[list[float], dict[date, float]]:
    name_to_factor = dict(zip(weekday_df["weekday"], weekday_df["factor"]))
    idx_to_name = weekday_index_map()
    uplift_map, elast_map = expand_event_ranges(events_df)

    adj = []
    for i, o in enumerate(base_occ):
        d = start + timedelta(days=i+1)  # forecast starts tomorrow
        wname = idx_to_name[d.weekday()]
        wfac = float(name_to_factor.get(wname, 1.0))
        ev = float(uplift_map.get(d, 1.0))
        val = float(np.clip(o * wfac * ev, 0.40, 0.98))
        adj.append(val)
    return adj, elast_map


def rate_recommendation(current_occ: float, room_rate: float, fences: tuple[float, float]) -> float:
    min_r, max_r = fences
    if current_occ < LOW_OCC_THRESHOLD:
        gap = (LOW_OCC_THRESHOLD - current_occ) / LOW_OCC_THRESHOLD
        adj = np.clip(gap * MAX_DOWN_ADJ, 0, MAX_DOWN_ADJ)
        rec = room_rate * (1 - adj)
    elif current_occ > HIGH_OCC_THRESHOLD:
        gap = (current_occ - HIGH_OCC_THRESHOLD) / (1 - HIGH_OCC_THRESHOLD)
        adj = np.clip(gap * MAX_UP_ADJ, 0, MAX_UP_ADJ)
        rec = room_rate * (1 + adj)
    else:
        rec = room_rate
    return float(np.clip(rec, min_r, max_r))


def elasticity_demand_multiplier(price_change_pct: float, elasticity: float) -> float:
    return max(0.0, 1.0 + elasticity * price_change_pct)


def allocate_by_capacity(demand_by_seg: Dict[str, float], capacity: int) -> Dict[str, float]:
    total = float(sum(demand_by_seg.values()))
    if total <= capacity or total == 0:
        return {k: min(v, capacity) for k, v in demand_by_seg.items()}
    scale = capacity / total
    return {k: v * scale for k, v in demand_by_seg.items()}


def compute_kpis(rooms_available: int, rooms_sold: float, revenue: float, revenue_net: float) -> Dict[str, float]:
    occ = 0 if rooms_available == 0 else rooms_sold / rooms_available
    adr = 0 if rooms_sold == 0 else revenue / rooms_sold
    revpar = 0 if rooms_available == 0 else revenue / rooms_available
    revpar_net = 0 if rooms_available == 0 else revenue_net / rooms_available
    return {
        "Occ%": occ,
        "ADR": adr,
        "RevPAR": revpar,
        "RevPAR_net": revpar_net,
        "Rooms Sold": rooms_sold,
        "Revenue": revenue,
        "Revenue_net": revenue_net
    }


def simulate_day(
    occ_unconst: float,
    room_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    today_rates_map: Dict[str, float],
    overbook: int,
    cancels_pct: float,
    elasticity_factor: float = 1.0,  # < 1.0 during compression
    occ_bias: float = 0.0,           # +/- adjust perceived occ for rate recs (budget pacing)
) -> dict:
    occ_unconst: float,
    room_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    today_rates_map: Dict[str, float],
    overbook: int,
    cancels_pct: float,
    elasticity_factor: float = 1.0,  # < 1.0 during compression
) -> dict:
    total_rooms = int(room_df["rooms"].sum())

    # Rate recommendation per type for this occupancy
    rec_rates = {}
    for _, r in room_df.iterrows():
        curr_rate = float(today_rates_map.get(r["room_type"], r["base        rec_rates[r["room_type"]] = rec

    # Weighted avg recommended ADR vs base
    base_avg = float(np.average(room_df["base_rate"], weights=room_df["rooms"]))
    rec_avg  = float(np.average([rec_rates[rt] for rt in room_df["room_type"]], weights=room_df["rooms"]))
    price_change_pct = 0 if base_avg == 0 else (rec_avg - base_avg) / base_avg

    # Unconstrained demand
    demand_rooms = occ_unconst * total_rooms

    # Segment demand and elasticity (dampened during events)
    demand_by_seg = {s["segment"]: demand_rooms * s["mix"] for _, s in seg_df.iterrows()}
    for _, s in seg_df.iterrows():
        eff_el = float(s["elasticity"]) * float(elasticity_factor)
        mult = elasticity_demand_multiplier(price_change_pct, eff_el)
        demand_by_seg[s["segment"]] *= mult

    # Capacity with overbooking; then reduce by cancels/no-shows
    capacity = total_rooms + overbook
    alloc = allocate_by_capacity(demand_by_seg, capacity)
    alloc = {k: v * (1 - cancels_pct) for k, v in alloc.items()}
    rooms_sold = float(sum(alloc.values()))

    # Revenue calc using average rec rate
    revenue_gross = rooms_sold * rec_avg

    # Net by commissions
    revenue_net = 0.0
    for _, s in seg_df.iterrows():
        seg_rooms = float(alloc[s["segment"]])
        seg_rev   = seg_rooms * rec_avg
        seg_net   = seg_rev * (1 - float(s["commission"]))
        revenue_net += seg_net

    kpis = compute_kpis(total_rooms, rooms_sold, revenue_gross, revenue_net)

    # Split by room type proportionally
    type_weights = room_df["rooms"] / room_df["rooms"].sum()
    by_type = pd.DataFrame({
        "room_type": room_df["room_type"],
        "rec_rate": [rec_rates[rt] for rt in room_df["room_type"]],
        "rooms_sold_est": rooms_sold * type_weights.values,
    })
    by_type["revenue_est"] = by_type["rooms_sold_est"] * by_type["rec_rate"]

    return {
        "rec_avg_rate": rec_avg,
        "price_change_pct": price_change_pct,
        "rooms_sold": rooms_sold,
        "revenue": revenue_gross,
        "revenue_net": revenue_net,
        "kpis": kpis,
        "by_type": by_type,
        "rec_rates": rec_rates,
    }

# -------------------------
# UI — Sidebar inputs
# -------------------------

st.title("🏨 Hotel Revenue Simulator — Ramada Abu Dhabi Corniche")
st.caption("Beginner-friendly tool for daily pricing, forecasting, and KPIs (with Abu Dhabi events)")

with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Room Types & Inventory", expanded=True):
        room_df = st.data_editor(
            DEFAULT_ROOMS.copy(),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "room_type": st.column_config.TextColumn("Room Type"),
                "rooms": st.column_config.NumberColumn("Rooms", min_value=0, step=1),
                "base_rate": st.column_config.NumberColumn("Base Rate", min_value=0.0, step=10.0),
                "min_rate": st.column_config.NumberColumn("Min Rate", min_value=0.0, step=10.0),
                "max_rate": st.column_config.NumberColumn("Max Rate", min_value=0.0, step=10.0),
            },
            key="rooms_editor",
        )

    with st.expander("Segments (Mix, Elasticity, Commission)", expanded=True):
        seg_df = st.data_editor(
            DEFAULT_SEGS.copy(),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "segment": st.column_config.TextColumn("Segment"),
                "mix": st.column_config.NumberColumn("Mix (sums≈1)", min_value=0.0, max_value=1.0, step=0.01),
                "elasticity": st.column_config.NumberColumn("Elasticity (neg)", step=0.05),
                "commission": st.column_config.NumberColumn("Commission %", min_value=0.0, max_value=1.0, step=0.01),
            },
            key="segs_editor",
        )
        seg_df = normalize_mix(seg_df)
        st.caption("Tip: Mix auto-normalized to sum = 1.0")

    st.divider()
    forecast_days = st.select_slider("Forecast horizon (days)", options=[7, 14, 21, 28], value=14)

    with st.expander("Weekday Factors (Mon–Sun)", expanded=False):
        weekday_df = st.data_editor(
            DEFAULT_WEEKDAY.copy(),
            use_container_width=True,
            num_rows=7,
            column_config={
                "weekday": st.column_config.TextColumn(disabled=True),
                "factor": st.column_config.NumberColumn("Factor", min_value=0.5, max_value=1.5, step=0.01),
            },
            key="weekday_editor",
        )

    with st.expander("Special Events (ranges, uplift & elasticity)", expanded=True):
        st.caption("Uplift >1 increases demand; Elasticity factor <1 makes demand less price-sensitive during events.")
        events_df = st.data_editor(
            PREPOP_EVENTS.copy(),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Event"),
                "start_date": st.column_config.DateColumn("Start Date"),
                "end_date": st.column_config.DateColumn("End Date"),
                "uplift_factor": st.column_config.NumberColumn("Uplift", min_value=0.8, max_value=2.5, step=0.01),
                "elasticity_factor": st.column_config.NumberColumn("Elasticity ×", min_value=0.2, max_value=1.2, step=0.05),
            },
            key="events_editor",
        )

    with st.expander("Budget & Pace (current month)", expanded=True):
        # Pick any date inside the budget month (defaults to today)
        budget_ref_date = st.date_input("Budget month (pick any date in the month)")
        # Initialize a budget table with segments present in seg_df
        _seg_names = seg_df["segment"].tolist() if not seg_df.empty else DEFAULT_SEGS["segment"].tolist()
        budget_init = pd.DataFrame({
            "segment": _seg_names,
            "budget_RN": [0]*len(_seg_names),
            "budget_ADR": [0.0]*len(_seg_names),
            "budget_Revenue": [0.0]*len(_seg_names),
            "MTD_actual_RN": [0]*len(_seg_names),
            "MTD_actual_Revenue": [0.0]*len(_seg_names),
        })
        budget_df = st.data_editor(
            budget_init,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "segment": st.column_config.TextColumn("Segment"),
                "budget_RN": st.column_config.NumberColumn("Budget RN (month)", min_value=0, step=1),
                "budget_ADR": st.column_config.NumberColumn("Budget ADR", min_value=0.0, step=1.0),
                "budget_Revenue": st.column_config.NumberColumn("Budget Revenue (optional)", min_value=0.0, step=100.0),
                "MTD_actual_RN": st.column_config.NumberColumn("MTD Actual RN (optional)", min_value=0, step=1),
                "MTD_actual_Revenue": st.column_config.NumberColumn("MTD Actual Revenue (optional)", min_value=0.0, step=100.0),
            },
            key="budget_editor",
        )
        st.caption("Tip: Fill at least Budget RN and ADR by segment. If you add MTD Actuals, the simulator will pace against remaining days.")

    st.divider()
    with st.expander("Overbooking & Cancels", expanded=False):
        overbook_rooms = st.number_input("Overbooking buffer (rooms)", min_value=0, max_value=100, value=DEFAULT_OVERBOOK)
        cancels_pct = st.number_input("Cancels/No-Show %", min_value=0.0, max_value=0.5, value=DEFAULT_CANCELS_NOSHOW, step=0.01, format="%.2f")
        cancels_pct = st.number_input("Cancels/No-Show %", min_value=0.0, max_value=0.5, value=DEFAULT_CANCELS_NOSHOW, step=0.01, format="%.2f")

# -------------------------
# UI — Main inputs
# -------------------------
colA, colB, colC = st.columns([1,1,1])
with colA:
    st.subheader("Today's Status")
    today_occ = st.slider("Today's Occupancy %", min_value=0, max_value=100, value=72, step=1) / 100.0
    today_adr = st.number_input("Today's ADR", min_value=0.0, value=340.0, step=10.0)

with colB:
    st.subheader("Today's Posted Rates")
    if room_df.empty:
        st.info("Add at least one room type in the sidebar.")
        st.stop()
    today_rates_df = pd.DataFrame({
        "room_type": room_df["room_type"],
        "today_rate": room_df["base_rate"],
    })
    today_rates_df = st.data_editor(
        today_rates_df,
        use_container_width=True,
        column_config={
            "room_type": st.column_config.TextColumn(disabled=True),
            "today_rate": st.column_config.NumberColumn("Rate", min_value=0.0, step=10.0),
        },
        key="rates_editor",
    )
    today_rates_map = dict(zip(today_rates_df["room_type"], today_rates_df["today_rate"]))

with colC:
    st.subheader("(Optional) Recent History")
    use_synth = st.toggle("Generate synthetic history", value=True)
    hist_days = st.number_input("History days (for MA)", min_value=3, max_value=30, value=14, step=1)
    occ_hist_text = ""
    adr_hist_text = ""
    if not use_synth:
        occ_hist_text = st.text_area("Occ history (comma% e.g. 68,72,74)")
        adr_hist_text = st.text_area("ADR history (comma e.g. 330,335,342)")

run_btn = st.button("🚀 Run Forecast & Recommendations", type="primary")

# -------------------------
# Execute simulation
# -------------------------
if run_btn:
    if room_df.empty or seg_df.empty:
        st.error("Please configure room types and segments.")
        st.stop()

    total_rooms = int(room_df["rooms"].sum())

    # Prepare history
    if use_synth:
        occ_hist, adr_hist = generate_synthetic_history(int(hist_days), today_occ, today_adr)
    else:
        try:
            occ_hist = [float(x.strip())/100.0 for x in occ_hist_text.split(",") if x.strip()]
            adr_hist = [float(x.strip()) for x in adr_hist_text.split(",") if x.strip()]
        except Exception:
            st.error("Could not parse history. Use numbers, separated by commas.")
            st.stop()
        if len(occ_hist) == 0 or len(adr_hist) == 0:
            st.error("Please provide non-empty history for both Occ and ADR or enable synthetic history.")
            st.stop()

    # Base occupancy forecast
    base_occ_forecast = moving_average_forecast(occ_hist, horizon=int(forecast_days), window=min(7, len(occ_hist)))
    start_date = date.today()

    # Apply weekday & events adjustments (also returns per-day elasticity factors)
    occ_forecast, elast_map = apply_weekday_and_events(base_occ_forecast, start=start_date, weekday_df=weekday_df, events_df=events_df)

    # Today's KPIs (from inputs)
    rooms_sold_today = round(today_occ * total_rooms)
    rev_today = today_adr * rooms_sold_today

    net_today = 0.0
    for _, s in seg_df.iterrows():
        seg_rooms = rooms_sold_today * float(s["mix"])
        seg_rev   = seg_rooms * today_adr
        seg_net   = seg_rev * (1 - float(s["commission"]))
        net_today += seg_net

    kpis_today = compute_kpis(total_rooms, rooms_sold_today, rev_today, net_today)

    # Rate recs for today (rule-based)
    rate_recs_today = []
    for _, r in room_df.iterrows():
        curr_rate = float(today_rates_map.get(r["room_type"], r["base_rate"]))
        rec = rate_recommendation(today_occ, curr_rate, (float(r["min_rate"]), float(r["max_rate"])) )
        rate_recs_today.append((r["room_type"], curr_rate, rec))
    rate_recs_df = pd.DataFrame(rate_recs_today, columns=["room_type", "today_rate", "recommended_rate"]) 

    # Forecast next N days
    projections = []
    rec_tables = []
    for i, occ_u in enumerate(occ_forecast, start=1):
        d = start_date + timedelta(days=i)
        sim = simulate_day(
            occ_unconst=float(occ_u),
            room_df=room_df,
            seg_df=seg_df,
            today_rates_map=today_rates_map,
            overbook=int(overbook_rooms),
            cancels_pct=float(cancels_pct),
            elasticity_factor=float(elast_map.get(d, 1.0)),
        )
        projections.append({
            "date": d,
            "Occ%": sim["kpis"]["Occ%"],
            "ADR": sim["kpis"]["ADR"],
            "RevPAR": sim["kpis"]["RevPAR"],
            "RevPAR_net": sim["kpis"]["RevPAR_net"],
            "Rooms Sold": sim["kpis"]["Rooms Sold"],
            "Revenue": sim["kpis"]["Revenue"],
            "Revenue_net": sim["kpis"]["Revenue_net"],
            "Rec_Avg_Rate": sim["rec_avg_rate"],
            "Price_Change_%": sim["price_change_pct"],
        })
        rec_row = {"date": d}
        rec_row.update(sim["rec_rates"])  # columns per room_type
        rec_tables.append(rec_row)

    forecast_df = pd.DataFrame(projections)
    rate_plan_future = pd.DataFrame(rec_tables)

    # -------------------------
    # OUTPUT — KPIs & Tables
    # -------------------------
    st.success("Done! See results below.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rooms", f"{total_rooms}")
    with col2:
        st.metric("Today Occ%", f"{kpis_today['Occ%']*100:,.1f}%")
    with col3:
        st.metric("Today ADR", f"{kpis_today['ADR']:,.0f}")
    with col4:
        st.metric("Today RevPAR (Net)", f"{kpis_today['RevPAR_net']:,.0f}")

    st.subheader("Today's Rate Recommendations")
    st.dataframe(rate_recs_df, use_container_width=True)

    st.subheader("Next Days Forecast (Summary)")
    dfp = forecast_df.copy()
    dfp["Occ%"] = (dfp["Occ%"]*100).round(1)
    dfp["Price_Change_%"] = (dfp["Price_Change_%"]*100).round(1)
    st.dataframe(dfp, use_container_width=True)

    # -------------------------
    # Charts
    # -------------------------
    st.subheader("Charts")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    # Occ
    ax[0].plot(dfp["date"], dfp["Occ%"], marker="o")
    ax[0].set_title("Forecast Occupancy %")
    ax[0].set_ylim(40, 100)
    ax[0].set_ylabel("%")

    # Rec Avg Rate
    ax[1].plot(dfp["date"], dfp["Rec_Avg_Rate"], color="orange", marker="o")
    ax[1].set_title("Recommended Avg Rate")
    ax[1].set_ylabel("Currency")

    # RevPAR
    ax[2].plot(dfp["date"], dfp["RevPAR"], color="green", marker="o")
    ax[2].set_title("RevPAR (Gross)")
    ax[2].set_ylabel("Currency")

    # Revenue
    ax[3].bar(dfp["date"], dfp["Revenue"], color="#4C72B0")
    ax[3].set_title("Total Revenue (Gross)")
    ax[3].set_ylabel("Currency")
    for a in ax:
        for label in a.get_xticklabels():
            label.set_rotation(45)

    st.pyplot(fig)

    # -------------------------
    # Downloads
    # -------------------------
    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Download forecast_summary.csv",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_summary.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Download today_rate_recommendations.csv",
            data=rate_recs_df.to_csv(index=False).encode("utf-8"),
            file_name="today_rate_recommendations.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            label="Download rate_plan_by_roomtype_future.csv",
            data=rate_plan_future.to_csv(index=False).encode("utf-8"),
            file_name="rate_plan_by_roomtype_future.csv",
            mime="text/csv",
        )

else:
    st.info("Configure inputs and click **Run Forecast & Recommendations**.")
