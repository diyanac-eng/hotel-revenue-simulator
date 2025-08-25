# Hotel Revenue Simulator — Streamlit App (Ramada Abu Dhabi Corniche)
# -----------------------------------------------------------------------------
# ✅ Focused on factual, user-entered inputs — no demand forecasting needed.
# This build prioritizes the *Inquiry Scenario* impact on today and a 7/14/28-day
# window, plus Budget/MTD variance. Seasonality/events modules can be disabled.
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
from datetime import date, datetime, timedelta
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
    elasticity: float    # not used when demand model off
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

# Segment setup (editable in UI)
DEFAULT_SEGS = pd.DataFrame([
    {"segment": "OTA",                 "mix": 0.2837, "elasticity": -0.80, "commission": 0.18},
    {"segment": "Walk-In",             "mix": 0.0386, "elasticity": -0.30, "commission": 0.00},
    {"segment": "Direct",              "mix": 0.0085, "elasticity": -0.50, "commission": 0.00},
    {"segment": "Discounted Retail-H", "mix": 0.0117, "elasticity": -0.90, "commission": 0.18},
    {"segment": "Corporate",           "mix": 0.4528, "elasticity": -0.20, "commission": 0.05},
    {"segment": "Tour Series",         "mix": 0.0764, "elasticity": -0.15, "commission": 0.00},
    {"segment": "Wholesaler",          "mix": 0.1271, "elasticity": -0.45, "commission": 0.15},
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

# (Kept in code but disabled by default)
DEFAULT_SEASON_FACTORS = {"High": 1.15, "Shoulder": 1.00, "Low": 0.85}
DEFAULT_MONTH_TO_SEASON = {1:"High",2:"High",3:"High",4:"High",5:"Shoulder",6:"Low",7:"Low",8:"Low",9:"Shoulder",10:"High",11:"High",12:"High"}
DEFAULT_SEASON_OVERRIDES = pd.DataFrame([
    {"start_date": pd.to_datetime("2026-03-01"), "end_date": pd.to_datetime("2026-03-31"), "factor": 1.00, "label": "Ramadan 2026 (shoulder)"}
])

# -------------------------
# Utility functions
# -------------------------

def normalize_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "mix" in df.columns and df["mix"].sum() > 0:
        df = df.copy(); df["mix"] = df["mix"] / df["mix"].sum()
    return df


def build_month_factor_map(season_factors: Dict[str, float], month_to_season: Dict[int, str]) -> Dict[int, float]:
    return {m: float(season_factors.get(season, 1.0)) for m, season in month_to_season.items()}


def expand_override_ranges(df: pd.DataFrame) -> Dict[date, float]:
    out: Dict[date, float] = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        try:
            sd = pd.to_datetime(r.get("start_date")).date(); ed = pd.to_datetime(r.get("end_date")).date(); fac = float(r.get("factor", 1.0))
            if sd is None or ed is None: continue
            d = sd
            while d <= ed:
                out[d] = fac; d += timedelta(days=1)
        except Exception:
            continue
    return out

# -------------------------
# UI — Sidebar inputs
# -------------------------
st.title("🏨 Hotel Revenue Simulator — Ramada Abu Dhabi Corniche")
st.caption("Factual impact from entered inquiries & rate strategy. No demand modeling required.")

with st.sidebar:
    st.header("⚙️ Settings")

    use_demand_modules = st.toggle("Enable demand modifiers (seasonality/events)?", value=False, help="Keep OFF to stay strictly factual.")

    with st.expander("Room Types & Inventory", expanded=True):
        room_df = st.data_editor(
            DEFAULT_ROOMS.copy(), use_container_width=True, num_rows="dynamic",
            column_config={
                "room_type": st.column_config.TextColumn("Room Type"),
                "rooms": st.column_config.NumberColumn("Rooms", min_value=0, step=1),
                "base_rate": st.column_config.NumberColumn("Base Rate", min_value=0.0, step=10.0),
                "min_rate": st.column_config.NumberColumn("Min Rate", min_value=0.0, step=10.0),
                "max_rate": st.column_config.NumberColumn("Max Rate", min_value=0.0, step=10.0),
            }, key="rooms_editor",
        )
        ooo = st.number_input("Out-of-order rooms (OOO)", min_value=0, max_value=int(room_df["rooms"].sum()), value=0)
        total_rooms = int(room_df["rooms"].sum())
        total_available = max(0, total_rooms - ooo)
        st.caption(f"Available tonight: **{total_available}** / {total_rooms}")

    with st.expander("Segments (Mix, Commission)", expanded=True):
        seg_df = st.data_editor(
            DEFAULT_SEGS.drop(columns=["elasticity"]).copy(), use_container_width=True, num_rows="dynamic",
            column_config={
                "segment": st.column_config.TextColumn("Segment"),
                "mix": st.column_config.NumberColumn("Mix (sums≈1)", min_value=0.0, max_value=1.0, step=0.01),
                "commission": st.column_config.NumberColumn("Commission %", min_value=0.0, max_value=1.0, step=0.01),
            }, key="segs_editor",
        )
        seg_df = normalize_mix(seg_df)

    with st.expander("Monthly Budget (segment RN & ADR) + MTD Actuals", expanded=True):
        today = date.today()
        days_in_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        dim = days_in_month.day; doy = today.day
        st.caption("Enter **full-month** budget by segment. We’ll compute MTD target = budget × (days passed / days in month).")
        default_bud = pd.DataFrame({
            "segment": seg_df["segment"],
            "budget_rn": 0, "budget_adr": 0.0,
            "mtd_actual_rn": 0, "mtd_actual_rev": 0.0,
        })
        budget_df = st.data_editor(
            default_bud, use_container_width=True, num_rows="dynamic",
            column_config={
                "segment": st.column_config.TextColumn(disabled=True),
                "budget_rn": st.column_config.NumberColumn("Budget RN (month)", min_value=0, step=10),
                "budget_adr": st.column_config.NumberColumn("Budget ADR (month)", min_value=0.0, step=5.0),
                "mtd_actual_rn": st.column_config.NumberColumn("MTD Actual RN", min_value=0, step=1),
                "mtd_actual_rev": st.column_config.NumberColumn("MTD Actual Rev", min_value=0.0, step=100.0),
            }, key="budget_editor",
        )
        st.caption(f"Today is day **{doy}** of **{dim}**.")

    with st.expander("Overbooking & Cancels", expanded=False):
        overbook_rooms = st.number_input("Overbooking buffer (rooms)", min_value=0, max_value=100, value=5)
        cancels_pct = st.number_input("Cancels/No-Show % (for info)", min_value=0.0, max_value=0.5, value=0.03, step=0.01, format="%.2f")

# -------------------------
# Main Inputs (Today + Posted Rates)
# -------------------------
colA, colB, colC = st.columns([1,1,1])
with colA:
    st.subheader("Today's Status")
    rooms_occupied_today = st.number_input("Rooms occupied today (absolute)", min_value=0, max_value=total_available, value=int(0.5*total_available))
    today_adr = st.number_input("Today's ADR (RO, excl. tax)", min_value=0.0, value=190.0, step=5.0)
    today_occ = 0.0 if total_available==0 else rooms_occupied_today/total_available

with colB:
    st.subheader("Today's Posted Rates (per room type)")
    today_rates_df = pd.DataFrame({"room_type": room_df["room_type"], "today_rate": room_df["base_rate"]})
    today_rates_df = st.data_editor(
        today_rates_df, use_container_width=True,
        column_config={"room_type": st.column_config.TextColumn(disabled=True),
                       "today_rate": st.column_config.NumberColumn("Rate", min_value=0.0, step=10.0)},
        key="rates_editor",
    )
    today_rates_map = dict(zip(today_rates_df["room_type"], today_rates_df["today_rate"]))

with colC:
    st.subheader("Baseline for Period (no forecast)")
    scenario_days_default = 7
    scenario_days = st.select_slider("Scenario period", options=[7,14,28], value=scenario_days_default)
    baseline_method = st.radio("Baseline method", ["Use today's KPIs (flat)", "Enter baseline per day"], index=0)
    scenario_start = st.date_input("Scenario start date", value=date.today())

# -------------------------
# Inquiry Scenario — Input table
# -------------------------
st.header("📋 Inquiry Scenario — Enter potential business")
scen_df = st.data_editor(
    pd.DataFrame([
        {"segment": "Corporate", "rooms_per_night": 0, "rate_ro": 0.0, "start_date": scenario_start, "nights": 1},
        {"segment": "OTA", "rooms_per_night": 0, "rate_ro": 0.0, "start_date": scenario_start, "nights": 1},
    ]),
    use_container_width=True, num_rows="dynamic",
    column_config={
        "segment": st.column_config.TextColumn("Segment"),
        "rooms_per_night": st.column_config.NumberColumn("Rooms/night", min_value=0, step=1),
        "rate_ro": st.column_config.NumberColumn("Rate (RO)", min_value=0.0, step=5.0),
        "start_date": st.column_config.DateColumn("Start"),
        "nights": st.column_config.NumberColumn("Nights", min_value=1, step=1),
    }, key="scenario_editor",
)

# Optional: Manual baseline table when selected
period_dates = [scenario_start + timedelta(days=i) for i in range(scenario_days)]
manual_base_df = pd.DataFrame({"date": period_dates, "base_rn": 0, "base_adr": today_adr})
if baseline_method == "Enter baseline per day":
    manual_base_df = st.data_editor(
        manual_base_df, use_container_width=True, num_rows="fixed",
        column_config={"date": st.column_config.DateColumn("Date"),
                       "base_rn": st.column_config.NumberColumn("Base Rooms Sold", min_value=0, step=1),
                       "base_adr": st.column_config.NumberColumn("Base ADR", min_value=0.0, step=5.0)},
        key="manual_base_editor",
    )

run_btn = st.button("✅ Run Calculation (No Forecast)", type="primary")

# -------------------------
# Execute Scenario Calculation (factual)
# -------------------------
if run_btn:
    # Build baseline maps
    base_rooms_map: Dict[date, float] = {}
    base_rev_map: Dict[date, float] = {}

    for d in period_dates:
        if baseline_method == "Use today's KPIs (flat)":
            base_rn = float(rooms_occupied_today) if d == date.today() else float(rooms_occupied_today)
            base_adr = float(today_adr)
        else:
            row = manual_base_df[manual_base_df["date"]==pd.to_datetime(d)].head(1)
            base_rn = float(row["base_rn"].iloc[0]) if not row.empty else 0.0
            base_adr = float(row["base_adr"].iloc[0]) if not row.empty else 0.0
        base_rooms_map[d] = base_rn
        base_rev_map[d] = base_rn * base_adr

    # Map of added rooms/revenue from inquiries
    add_rooms_map: Dict[date, float] = {d: 0.0 for d in period_dates}
    add_rev_map: Dict[date, float] = {d: 0.0 for d in period_dates}
    total_requested_roomnights = 0.0

    for _, row in scen_df.iterrows():
        seg = str(row.get("segment", ""))
        rooms_per_night = float(row.get("rooms_per_night", 0) or 0)
        rate_ro = float(row.get("rate_ro", 0) or 0)
        sd = pd.to_datetime(row.get("start_date", scenario_start)).date()
        n = int(row.get("nights", 1) or 1)
        for i in range(n):
            d = sd + timedelta(days=i)
            if d in add_rooms_map:
                add_rooms_map[d] += rooms_per_night
                add_rev_map[d] += rooms_per_night * rate_ro
                total_requested_roomnights += rooms_per_night

    # Capacity cap per day
    cap = total_available + overbook_rooms

    # Recompute daily KPIs with capacity cap
    scenario_rows = []
    total_spill = 0.0
    for d in period_dates:
        base_rn = base_rooms_map[d]
        base_rev = base_rev_map[d]
        add_rn = add_rooms_map[d]
        add_rev = add_rev_map[d]

        uncapped = base_rn + add_rn
        spill = max(0.0, uncapped - cap)
        keep_add_rn = max(0.0, min(add_rn, cap - base_rn))
        keep_add_rev = 0.0 if add_rn == 0 else add_rev * (keep_add_rn / add_rn)
        new_rn = min(cap, uncapped)
        new_rev = base_rev + keep_add_rev

        new_occ = 0.0 if total_available == 0 else new_rn / total_available
        new_adr = 0.0 if new_rn == 0 else new_rev / new_rn
        new_revpar = 0.0 if total_available == 0 else new_rev / total_available

        scenario_rows.append({
            "date": d,
            "Base RN": base_rn,
            "Base ADR": 0.0 if base_rn == 0 else base_rev / base_rn,
            "Added RN": keep_add_rn,
            "Added Rev": keep_add_rev,
            "New RN": new_rn,
            "New ADR": new_adr,
            "New Occ%": new_occ*100,
            "New RevPAR": new_revpar,
            "Spill RN": spill,
        })
        total_spill += spill

    scenario_df = pd.DataFrame(scenario_rows)

    # --- Today impact
    today_row = scenario_df.loc[scenario_df["date"] == date.today()]
    st.subheader("Today Impact (factual)")
    if not today_row.empty:
        t = today_row.iloc[0]
        st.markdown(f"**Occ {t['New Occ%']:.1f}% · ADR {t['New ADR']:,.0f} · RevPAR {t['New RevPAR']:,.0f} · Rooms {t['New RN']:,.0f} (ΔRN +{t['Added RN']:,.0f})**")
    else:
        st.info("Scenario does not include today. Set start date to today to see same-day impact.")

    # --- Period summary + ALOS for new business
    period_new_rev = float(scenario_df["New RN"].mul(scenario_df["New ADR"]).sum())
    period_new_rn  = float(scenario_df["New RN"].sum())
    period_new_occ = 0.0 if total_available == 0 else period_new_rn / (total_available * len(period_dates))
    period_new_adr = 0.0 if period_new_rn == 0 else period_new_rev / period_new_rn
    period_new_revpar = 0.0 if total_available == 0 else period_new_rev / (total_available * len(period_dates))

    # ALOS estimation for inquiries
    try:
        mean_nights = float(np.mean([int(n) for n in scen_df["nights"].fillna(1).tolist()]))
    except Exception:
        mean_nights = 1.0
    total_inquiry_rn = sum(add_rooms_map.values())
    total_inquiry_stays = max(1.0, total_inquiry_rn / max(1.0, mean_nights))
    alos_inquiries = 0.0 if total_inquiry_stays == 0 else total_inquiry_rn / total_inquiry_stays

    st.subheader("Period Summary")
    cA, cB, cC, cD = st.columns(4)
    with cA: st.metric("Period Occ%", f"{period_new_occ*100:,.1f}%")
    with cB: st.metric("Period ADR", f"{period_new_adr:,.0f}")
    with cC: st.metric("Period RevPAR", f"{period_new_revpar:,.0f}")
    with cD: st.metric("Spill (RN)", f"{total_spill:,.0f}")

    st.dataframe(scenario_df, use_container_width=True)
    st.caption(f"ALOS for new inquiries (est.): {alos_inquiries:,.2f} nights. Set exact Nights per line above for precise ALOS.")

    # -------------------------
    # Budget / MTD impact (factual pacing)
    # -------------------------
    st.subheader("Budget / MTD Impact")
    today = date.today()
    month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    dim = month_end.day; doy = today.day

    bud = budget_df.copy()
    if not bud.empty and dim > 0:
        bud["target_mtd_rn"] = bud["budget_rn"].astype(float) * (doy / dim)
        bud["target_mtd_rev"] = bud["target_mtd_rn"] * bud["budget_adr"].astype(float)
    else:
        bud["target_mtd_rn"] = 0.0; bud["target_mtd_rev"] = 0.0

    # Allocate inquiry RN/Rev by segment for dates that fall within MTD (today only adds to MTD)
    # We don’t forecast other days; we only add scenario dates that are <= today (factual) OR optionally include to-go for visibility.
    include_to_go = st.toggle("Also show projected month-end *increment* from scenario (adds future scenario dates only)", value=True)

    # Build segment-level increments from scenario
    seg_increment_mtd = {s: {"rn":0.0, "rev":0.0} for s in seg_df["segment"].tolist()}
    seg_increment_to_go = {s: {"rn":0.0, "rev":0.0} for s in seg_df["segment"].tolist()}

    for _, row in scen_df.iterrows():
        seg = str(row.get("segment", ""))
        rooms_per_night = float(row.get("rooms_per_night", 0) or 0)
        rate_ro = float(row.get("rate_ro", 0) or 0)
        sd = pd.to_datetime(row.get("start_date", scenario_start)).date()
        n = int(row.get("nights", 1) or 1)
        for i in range(n):
            d = sd + timedelta(days=i)
            if d.month == today.month and d.year == today.year:
                if d <= today:
                    seg_increment_mtd.setdefault(seg, {"rn":0.0, "rev":0.0})
                    seg_increment_mtd[seg]["rn"] += rooms_per_night
                    seg_increment_mtd[seg]["rev"] += rooms_per_night * rate_ro
                elif include_to_go:
                    seg_increment_to_go.setdefault(seg, {"rn":0.0, "rev":0.0})
                    seg_increment_to_go[seg]["rn"] += rooms_per_night
                    seg_increment_to_go[seg]["rev"] += rooms_per_night * rate_ro

    inc_df_rows = []
    for seg in seg_df["segment"].tolist():
        mtd_rn = float(seg_increment_mtd.get(seg, {}).get("rn", 0.0))
        mtd_rev = float(seg_increment_mtd.get(seg, {}).get("rev", 0.0))
        go_rn = float(seg_increment_to_go.get(seg, {}).get("rn", 0.0))
        go_rev = float(seg_increment_to_go.get(seg, {}).get("rev", 0.0))
        inc_df_rows.append({"segment": seg, "scenario_MTD_add_RN": mtd_rn, "scenario_MTD_add_Rev": mtd_rev, "scenario_ToGo_add_RN": go_rn, "scenario_ToGo_add_Rev": go_rev})
    inc_df = pd.DataFrame(inc_df_rows)

    # Combine with budget + actuals
    bud = bud.merge(inc_df, on="segment", how="left").fillna(0)
    bud["mtd_after_rn"] = bud["mtd_actual_rn"] + bud["scenario_MTD_add_RN"]
    bud["mtd_after_rev"] = bud["mtd_actual_rev"] + bud["scenario_MTD_add_Rev"]
    bud["mtd_var_rn"] = bud["mtd_after_rn"] - bud["target_mtd_rn"]
    bud["mtd_var_rev"] = bud["mtd_after_rev"] - bud["target_mtd_rev"]
    bud["mtd_var_rn_%"] = np.where(bud["target_mtd_rn"]>0, bud["mtd_var_rn"]/bud["target_mtd_rn"], 0.0)
    bud["mtd_var_rev_%"] = np.where(bud["target_mtd_rev"]>0, bud["mtd_var_rev"]/bud["target_mtd_rev"], 0.0)

    # Optional month-end projection delta from scenario (no other pickup assumed)
    bud["to_go_add_rn"] = bud["scenario_ToGo_add_RN"]
    bud["to_go_add_rev"] = bud["scenario_ToGo_add_Rev"]

    st.dataframe(bud[["segment","target_mtd_rn","mtd_actual_rn","scenario_MTD_add_RN","mtd_after_rn","mtd_var_rn","mtd_var_rn_%",
                      "target_mtd_rev","mtd_actual_rev","scenario_MTD_add_Rev","mtd_after_rev","mtd_var_rev","mtd_var_rev_%",
                      "to_go_add_rn","to_go_add_rev"]], use_container_width=True)

    st.info("Budget math: Target MTD = (Budget month) × (days passed / days in month). We only add scenario RN/Rev on dates you entered (no forecasts).")

    # -------------------------
    # Simple visuals
    # -------------------------
    st.subheader("Charts")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Today before/after bars
    trow = scenario_df.loc[scenario_df["date"] == date.today()]
    if not trow.empty:
        t = trow.iloc[0]
        ax[0].bar(["Before","After"], [t["Base RN"], t["New RN"]], color=["#bbbbbb","#4C72B0"])
        ax[0].set_title("Today Rooms Sold")
        ax[0].set_ylabel("Rooms")
        ax[1].bar(["Before","After"], [t["Base ADR"], t["New ADR"]], color=["#bbbbbb","#FFA500"])
        ax[1].set_title("Today ADR (RO)")
        ax[1].set_ylabel("ADR")
        st.pyplot(fig)
    else:
        ax[0].axis('off'); ax[1].axis('off'); st.pyplot(fig)

else:
    st.info("Enter today's figures and your inquiries, then click **Run Calculation (No Forecast)**.")
