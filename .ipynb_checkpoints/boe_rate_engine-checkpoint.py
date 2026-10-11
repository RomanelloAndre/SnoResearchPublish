# -*- coding: utf-8 -*-
"""
BoE Implied Bank Rate & Probability Engine
Bootstrapped from Quarterly IMM 3-Month SONIA Futures (ICE: SO3 / SONIA.N)
via Regularized Weighted Linear Least Squares (WLS).
Automated ingestion from data/raw_rates.csv and export to data/*.csv.
"""

import os
import datetime
import math
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear

# ==============================================================================
# 1. GLOBAL INPUTS & OFFICIAL MPC CALENDAR
# ==============================================================================
CURRENT_BANK_RATE = 3.75   # Official Bank of England Bank Rate (%)
SPREAD_BPS = 0.03          # SONIA fixes ~3 bps below Bank Rate (added to futures yield)
SMOOTHNESS_LAMBDA = 1e-3   # Regularization penalty to prevent step-function spikes

# Standard Quarterly IMM Reference Windows (3rd Wednesday of Mar/Jun/Sep/Dec)
THREE_MONTH_CONTRACTS = {
    "202609": {"start": datetime.date(2026, 9, 16),  "end": datetime.date(2026, 12, 16), "code": "SO3U6"},
    "202612": {"start": datetime.date(2026, 12, 16), "end": datetime.date(2027, 3, 17),  "code": "SO3Z6"},
    "202703": {"start": datetime.date(2027, 3, 17),  "end": datetime.date(2027, 6, 16),  "code": "SO3H7"},
    "202706": {"start": datetime.date(2027, 6, 16),  "end": datetime.date(2027, 9, 15),  "code": "SO3M7"},
    "202709": {"start": datetime.date(2027, 9, 15),  "end": datetime.date(2027, 12, 15), "code": "SO3U7"},
    "202712": {"start": datetime.date(2027, 12, 15), "end": datetime.date(2028, 3, 15),  "code": "SO3Z7"},
    "202803": {"start": datetime.date(2028, 3, 15),  "end": datetime.date(2028, 6, 21),  "code": "SO3H8"},
}

# Official Bank of England MPC Announcement Dates (Effective upon announcement)
BOE_MEETINGS = [
    {"name": "BoE Meeting: Nov 05, 2026", "meeting_date": datetime.date(2026, 11, 5),  "effective_date": datetime.date(2026, 11, 5)},
    {"name": "BoE Meeting: Dec 17, 2026", "meeting_date": datetime.date(2026, 12, 17), "effective_date": datetime.date(2026, 12, 17)},
    {"name": "BoE Meeting: Feb 04, 2027", "meeting_date": datetime.date(2027, 2, 4),   "effective_date": datetime.date(2027, 2, 4)},
    {"name": "BoE Meeting: Mar 18, 2027", "meeting_date": datetime.date(2027, 3, 18),  "effective_date": datetime.date(2027, 3, 18)},
    {"name": "BoE Meeting: Apr 29, 2027", "meeting_date": datetime.date(2027, 4, 29),  "effective_date": datetime.date(2027, 4, 29)},
    {"name": "BoE Meeting: Jun 17, 2027", "meeting_date": datetime.date(2027, 6, 17),  "effective_date": datetime.date(2027, 6, 17)},
    {"name": "BoE Meeting: Jul 29, 2027", "meeting_date": datetime.date(2027, 7, 29),  "effective_date": datetime.date(2027, 7, 29)},
    {"name": "BoE Meeting: Sep 16, 2027", "meeting_date": datetime.date(2027, 9, 16),  "effective_date": datetime.date(2027, 9, 16)},
    {"name": "BoE Meeting: Nov 04, 2027", "meeting_date": datetime.date(2027, 11, 4),  "effective_date": datetime.date(2027, 11, 4)},
    {"name": "BoE Meeting: Dec 16, 2027", "meeting_date": datetime.date(2027, 12, 16), "effective_date": datetime.date(2027, 12, 16)},
    {"name": "BoE Meeting: Feb 03, 2028", "meeting_date": datetime.date(2028, 2, 3),   "effective_date": datetime.date(2028, 2, 3)},
    {"name": "BoE Meeting: Mar 16, 2028", "meeting_date": datetime.date(2028, 3, 16),  "effective_date": datetime.date(2028, 3, 16)},
]

# ==============================================================================
# 2. AUTOMATED DATA LOADER
# ==============================================================================
def load_boe_futures(csv_path="data/raw_rates.csv"):
    """
    Reads harvested futures quotes from raw_rates.csv and extracts settlement
    prices for Bank of England (SONIA) contracts.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found at: {csv_path}. Run fetch_rates.py first.")

    df = pd.read_csv(csv_path)
    df_boe = df[df["central_bank"].str.upper() == "BOE"].copy()

    if df_boe.empty:
        raise ValueError("No records found for BOE in raw_rates.csv.")

    prices = {}
    as_of_date = str(df_boe["date"].iloc[-1])

    for _, row in df_boe.iterrows():
        month_str = str(row["contract_month"]).strip()[:6]
        settle = float(row["settle_price"])
        prices[month_str] = settle

        if "local_symbol" in row and pd.notna(row["local_symbol"]):
            prices[str(row["local_symbol"]).strip()] = settle

    print(f"Loaded {len(df_boe)} BoE futures quotes from {csv_path} (As-of Date: {as_of_date})")
    return prices, as_of_date

# ==============================================================================
# 3. GLOBAL CURVE BOOTSTRAPPER (Regularized WLS Optimization)
# ==============================================================================
def bootstrap_3m_sonia_curve(current_bank_rate, spread_bps, prices, as_of_date=None, smoothness_lambda=SMOOTHNESS_LAMBDA):
    """
    Solves A * x = b via Bounded Linear Least Squares with smoothing regularization.
    Resolves multi-meeting quarterly overlaps without arbitrary delta allocations.
    """
    if as_of_date:
        today = datetime.datetime.strptime(as_of_date, "%Y-%m-%d").date()
    else:
        today = datetime.date.today()

    future_meetings = [m for m in BOE_MEETINGS if m["effective_date"] >= today]
    
    valid_contracts = {}
    matched_prices = {}
    for month_key, bounds in THREE_MONTH_CONTRACTS.items():
        if month_key in prices:
            valid_contracts[month_key] = bounds
            matched_prices[month_key] = prices[month_key]
        elif bounds["code"] in prices:
            valid_contracts[month_key] = bounds
            matched_prices[month_key] = prices[bounds["code"]]

    if not valid_contracts or not future_meetings:
        print("⚠️ Warning: No valid overlapping contracts or future meetings found.")
        return []

    m1_date = future_meetings[0]["effective_date"]
    active_contracts = {k: v for k, v in valid_contracts.items() if v["end"] > m1_date}

    if not active_contracts:
        print("⚠️ Warning: No active contracts covering post-meeting periods.")
        return []

    max_date = max(c["end"] for c in active_contracts.values())
    relevant_meetings = [m for m in future_meetings if m["effective_date"] < max_date]

    n_contracts = len(active_contracts)
    n_meetings = len(relevant_meetings)

    if n_meetings == 0:
        return []

    A = np.zeros((n_contracts, n_meetings))
    b = np.zeros(n_contracts)
    meeting_eff_dates = [m["effective_date"] for m in relevant_meetings]

    for row_idx, (code, bounds) in enumerate(active_contracts.items()):
        c_start = bounds["start"]
        c_end = bounds["end"]
        total_days = (c_end - c_start).days
        actual_yield = (100.0 - matched_prices[code]) + spread_bps

        # Elapsed days prior to next rate decision (priced at current Bank Rate)
        overlap_r0 = max(0, (min(c_end, m1_date) - c_start).days)

        # Fraction of contract days spent in each post-meeting rate regime
        for col_idx in range(n_meetings):
            eff_start = meeting_eff_dates[col_idx]
            eff_end = meeting_eff_dates[col_idx + 1] if (col_idx + 1 < n_meetings) else datetime.date.max

            overlap_start = max(c_start, eff_start)
            overlap_end = min(c_end, eff_end)
            days_at_rate = max(0, (overlap_end - overlap_start).days)
            A[row_idx, col_idx] = days_at_rate / total_days

        b[row_idx] = actual_yield - (overlap_r0 / total_days) * current_bank_rate

    # Regularization: Penalizes abrupt step jumps between adjacent meetings in same quarter
    if n_meetings > 1 and smoothness_lambda > 0:
        D = np.zeros((n_meetings - 1, n_meetings))
        for i in range(n_meetings - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        reg_weight = math.sqrt(smoothness_lambda)
        A_aug = np.vstack([A, reg_weight * D])
        b_aug = np.concatenate([b, np.zeros(n_meetings - 1)])
    else:
        A_aug = A
        b_aug = b

    # Bound BoE Bank Rate between 0.00% and 15.00%
    res = lsq_linear(A_aug, b_aug, bounds=(np.zeros(n_meetings), np.full(n_meetings, 15.0)))

    implied_schedule = []
    for i, m in enumerate(relevant_meetings):
        implied_schedule.append({
            "meeting_name": m["name"],
            "meeting_date": str(m["meeting_date"]),
            "implied_rate": round(float(res.x[i]), 3)
        })

    return implied_schedule

# ==============================================================================
# 4. PROBABILITY ENGINE (Markov Chain Distribution)
# ==============================================================================
def calculate_probabilities(current_bank_rate, implied_schedule):
    """
    Computes step probabilities for 25 bps moves and builds Markov probability trees.
    """
    state_probs = {current_bank_rate: 1.0}
    prev_rate = current_bank_rate
    all_meeting_records = []

    for mtg in implied_schedule:
        mtg_name = mtg["meeting_name"]
        curr_implied_rate = mtg["implied_rate"]

        delta_rate = curr_implied_rate - prev_rate
        steps = (delta_rate * 100) / 25.0
        k = math.floor(steps)
        r = steps - k

        prob_step_lower = 1.0 - r
        prob_step_upper = r

        next_state_probs = {}
        for prev_level, prev_prob in state_probs.items():
            target_lower = round(prev_level + (k * 0.25), 2)
            target_upper = round(prev_level + ((k + 1) * 0.25), 2)

            next_state_probs[target_lower] = next_state_probs.get(target_lower, 0.0) + (prev_prob * prob_step_lower)
            next_state_probs[target_upper] = next_state_probs.get(target_upper, 0.0) + (prev_prob * prob_step_upper)

        for rate_val, prob in sorted(next_state_probs.items()):
            prob_pct = round(prob * 100, 2)
            if prob_pct > 0.5:  # Filter out trace probabilities (<0.5%)
                all_meeting_records.append({
                    "meeting_name": mtg_name,
                    "target_rate": f"{rate_val:.2f}%",
                    "rate_numeric": rate_val,
                    "probability": prob_pct,
                    "implied_rate": curr_implied_rate
                })

        state_probs = next_state_probs
        prev_rate = curr_implied_rate

    df_probs = pd.DataFrame(all_meeting_records)
    return df_probs

# ==============================================================================
# 5. PIPELINE EXECUTION & CSV EXPORT
# ==============================================================================
def run_boe_engine():
    prices, as_of = load_boe_futures("data/raw_rates.csv")
    schedule = bootstrap_3m_sonia_curve(
        current_bank_rate=CURRENT_BANK_RATE,
        spread_bps=SPREAD_BPS,
        prices=prices,
        as_of_date=as_of
    )

    if not schedule:
        print("⚠️ Could not generate implied rate schedule from current raw rates.")
        return None, None

    # 1. Export Implied Curve CSV
    df_curve = pd.DataFrame(schedule)
    df_curve["as_of_date"] = as_of
    curve_output = "data/boe_implied_curve.csv"
    df_curve.to_csv(curve_output, index=False)
    print(f"✓ Saved BoE implied curve to {curve_output}")

    # 2. Export Probabilities CSV
    df_probs = calculate_probabilities(CURRENT_BANK_RATE, schedule)
    df_probs["as_of_date"] = as_of
    probs_output = "data/boe_probabilities.csv"
    df_probs.to_csv(probs_output, index=False)
    print(f"✓ Saved BoE probability distribution to {probs_output}")

    # Display console previews
    print("\n--- IMPLIED BOE POLICY RATE SCHEDULE ---")
    print(df_curve[["meeting_name", "meeting_date", "implied_rate"]].to_string(index=False))

    print("\n--- TARGET PROBABILITY MATRIX (%) ---")
    df_pivot = df_probs.pivot(index="target_rate", columns="meeting_name", values="probability").fillna(0.0)
    print(df_pivot)

    return df_curve, df_probs

if __name__ == "__main__":
    run_boe_engine()