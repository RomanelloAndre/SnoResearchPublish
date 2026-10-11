# -*- coding: utf-8 -*-
"""
Reserve Bank of Australia (RBA) Implied Cash Rate & Probability Engine
Bootstrapped from ASX 30-Day Interbank Cash Rate Futures (SNFE / ASX: IB / IQ)
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
# 1. GLOBAL INPUTS & OFFICIAL RBA BOARD CALENDAR
# ==============================================================================
CURRENT_RBA_RATE = 4.60   # Current RBA Target Cash Rate (%)
SPREAD_BPS = 0.00         # 30-Day Futures track Cash Rate target directly (0.00% spread)
SMOOTHNESS_LAMBDA = 1e-3  # Regularization penalty to prevent step-function oscillation

# ASX 30-Day Interbank Cash Rate Futures Monthly Windows
ONE_MONTH_CONTRACTS = {
    "202609": {"start": datetime.date(2026, 9, 1),   "end": datetime.date(2026, 10, 1),  "code": "IBU6", "alt": "IQU26"},
    "202610": {"start": datetime.date(2026, 10, 1),  "end": datetime.date(2026, 11, 1),  "code": "IBV6", "alt": "IQV26"},
    "202611": {"start": datetime.date(2026, 11, 1),  "end": datetime.date(2026, 12, 1),  "code": "IBX6", "alt": "IQX26"},
    "202612": {"start": datetime.date(2026, 12, 1),  "end": datetime.date(2027, 1, 1),   "code": "IBZ6", "alt": "IQZ26"},
    "202701": {"start": datetime.date(2027, 1, 1),   "end": datetime.date(2027, 2, 1),   "code": "IBF7", "alt": "IQF27"},
    "202702": {"start": datetime.date(2027, 2, 1),   "end": datetime.date(2027, 3, 1),   "code": "IBG7", "alt": "IQG27"},
    "202703": {"start": datetime.date(2027, 3, 1),   "end": datetime.date(2027, 4, 1),   "code": "IBH7", "alt": "IQH27"},
    "202704": {"start": datetime.date(2027, 4, 1),   "end": datetime.date(2027, 5, 1),   "code": "IBJ7", "alt": "IQJ27"},
    "202705": {"start": datetime.date(2027, 5, 1),   "end": datetime.date(2027, 6, 1),   "code": "IBK7", "alt": "IQK27"},
    "202706": {"start": datetime.date(2027, 6, 1),   "end": datetime.date(2027, 7, 1),   "code": "IBM7", "alt": "IQM27"},
    "202707": {"start": datetime.date(2027, 7, 1),   "end": datetime.date(2027, 8, 1),   "code": "IBN7", "alt": "IQN27"},
    "202708": {"start": datetime.date(2027, 8, 1),   "end": datetime.date(2027, 9, 1),   "code": "IBQ7", "alt": "IQQ27"},
    "202709": {"start": datetime.date(2027, 9, 1),   "end": datetime.date(2027, 10, 1),  "code": "IBU7", "alt": "IQU27"},
}

# Official RBA Monetary Policy Board Decision Dates
# Rate changes take effect the day following the announcement (Wednesday)
RBA_MEETINGS = [
    {"name": "RBA Meeting: Nov 03, 2026", "meeting_date": datetime.date(2026, 11, 3), "effective_date": datetime.date(2026, 11, 4)},
    {"name": "RBA Meeting: Dec 08, 2026", "meeting_date": datetime.date(2026, 12, 8), "effective_date": datetime.date(2026, 12, 9)},
    {"name": "RBA Meeting: Feb 09, 2027", "meeting_date": datetime.date(2027, 2, 9),  "effective_date": datetime.date(2027, 2, 10)},
    {"name": "RBA Meeting: Mar 23, 2027", "meeting_date": datetime.date(2027, 3, 23), "effective_date": datetime.date(2027, 3, 24)},
    {"name": "RBA Meeting: May 04, 2027", "meeting_date": datetime.date(2027, 5, 4),  "effective_date": datetime.date(2027, 5, 5)},
    {"name": "RBA Meeting: Jun 22, 2027", "meeting_date": datetime.date(2027, 6, 22), "effective_date": datetime.date(2027, 6, 23)},
    {"name": "RBA Meeting: Aug 10, 2027", "meeting_date": datetime.date(2027, 8, 10), "effective_date": datetime.date(2027, 8, 11)},
    {"name": "RBA Meeting: Sep 28, 2027", "meeting_date": datetime.date(2027, 9, 28), "effective_date": datetime.date(2027, 9, 29)},
]

# ==============================================================================
# 2. AUTOMATED DATA LOADER
# ==============================================================================
def load_rba_futures(csv_path="data/raw_rates.csv"):
    """
    Reads harvested futures quotes from raw_rates.csv and extracts settlement
    prices for RBA 30-Day Interbank Cash Rate contracts.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found at: {csv_path}. Run fetch_rates.py first.")

    df = pd.read_csv(csv_path)
    df_rba = df[df["central_bank"].str.upper() == "RBA"].copy()

    if df_rba.empty:
        raise ValueError("No records found for RBA in raw_rates.csv.")

    prices = {}
    as_of_date = str(df_rba["date"].iloc[-1])

    for _, row in df_rba.iterrows():
        month_str = str(row["contract_month"]).strip()[:6]
        settle = float(row["settle_price"])
        prices[month_str] = settle

        if "local_symbol" in row and pd.notna(row["local_symbol"]):
            prices[str(row["local_symbol"]).strip()] = settle

    print(f"Loaded {len(df_rba)} RBA futures quotes from {csv_path} (As-of Date: {as_of_date})")
    return prices, as_of_date

# ==============================================================================
# 3. GLOBAL CURVE BOOTSTRAPPER (Regularized WLS Optimization)
# ==============================================================================
def bootstrap_1m_rba_curve(current_rba_rate, spread_bps, prices, as_of_date=None, smoothness_lambda=SMOOTHNESS_LAMBDA, weights=None):
    """
    Solves A * x = b via Bounded Linear Least Squares with smoothing regularization.
    Resolves intra-month meeting step shifts without non-linear solver divergence.
    """
    if as_of_date:
        today = datetime.datetime.strptime(as_of_date, "%Y-%m-%d").date()
    else:
        today = datetime.date.today()

    future_meetings = [m for m in RBA_MEETINGS if m["effective_date"] >= today]

    # Match contracts against ingested prices (by month code, localSymbol, or alt ticker)
    valid_contracts = {}
    matched_prices = {}
    for month_key, bounds in ONE_MONTH_CONTRACTS.items():
        if month_key in prices:
            valid_contracts[month_key] = bounds
            matched_prices[month_key] = prices[month_key]
        elif bounds["code"] in prices:
            valid_contracts[month_key] = bounds
            matched_prices[month_key] = prices[bounds["code"]]
        elif bounds.get("alt") in prices:
            valid_contracts[month_key] = bounds
            matched_prices[month_key] = prices[bounds["alt"]]

    if not valid_contracts or not future_meetings:
        print("⚠️ Warning: No valid overlapping contracts or future meetings found.")
        return []

    first_eff_date = future_meetings[0]["effective_date"]
    active_contracts = {k: v for k, v in valid_contracts.items() if v["end"] > first_eff_date}

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
        actual_yield = (100.0 - matched_prices[code]) + (spread_bps / 100.0)

        # Baseline days prior to next rate decision (priced at current Cash Rate)
        overlap_r0 = max(0, (min(c_end, first_eff_date) - c_start).days)

        # Fraction of contract days spent in each post-meeting rate regime
        for col_idx in range(n_meetings):
            eff_start = meeting_eff_dates[col_idx]
            eff_end = meeting_eff_dates[col_idx + 1] if (col_idx + 1 < n_meetings) else datetime.date.max

            overlap_start = max(c_start, eff_start)
            overlap_end = min(c_end, eff_end)
            days_at_rate = max(0, (overlap_end - overlap_start).days)
            A[row_idx, col_idx] = days_at_rate / total_days

        b[row_idx] = actual_yield - (overlap_r0 / total_days) * current_rba_rate

    # Optional Liquidity Weighting
    if weights is not None:
        w_vec = np.array([math.sqrt(weights.get(k, 1.0)) for k in active_contracts.keys()])
        W = np.diag(w_vec)
        A = W @ A
        b = W @ b

    # Policy Inertia Penalty: minimizes (r_{t+1} - r_t)^2
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

    # Bound RBA Target Cash Rate between 0.00% and 15.00%
    res = lsq_linear(A_aug, b_aug, bounds=(np.zeros(n_meetings), np.full(n_meetings, 15.0)))

    implied_schedule = []
    for i, m in enumerate(relevant_meetings):
        implied_schedule.append({
            "meeting_name": m["name"],
            "meeting_date": str(m["meeting_date"]),
            "effective_date": str(m["effective_date"]),
            "implied_rate": round(float(res.x[i]), 3)
        })

    return implied_schedule

# ==============================================================================
# 4. PROBABILITY ENGINE (Markov Chain Distribution)
# ==============================================================================
def calculate_probabilities(current_rba_rate, implied_schedule):
    """
    Computes step probabilities for 25 bps moves and builds Markov probability trees.
    """
    state_probs = {current_rba_rate: 1.0}
    prev_rate = current_rba_rate
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
            if prob_pct > 0.5:  # Filter trace probabilities
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
# 5. VISUALIZATION GENERATORS
# ==============================================================================
def plot_master_tree(df_probs):
    """Generates grouped probability chart across all upcoming RBA meetings."""
    fig = px.bar(
        df_probs,
        x="target_rate",
        y="probability",
        color="meeting_name",
        barmode="group",
        text="probability",
        title="<b>Implied RBA Cash Rate Target Probability Distribution (30-Day Futures)</b>",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 105],
        template="plotly_white",
        height=500,
        xaxis_title="RBA Cash Rate Target",
        yaxis_title="Probability (%)",
        legend_title="Meeting Date"
    )
    return fig

def plot_implied_curve(implied_schedule):
    """Generates the implied policy path line chart."""
    df_curve = pd.DataFrame(implied_schedule)
    df_curve['meeting_date'] = pd.to_datetime(df_curve['meeting_date'])

    fig = px.line(
        df_curve,
        x='meeting_date',
        y='implied_rate',
        markers=True,
        title='<b>Implied RBA Cash Rate Path (30-Day Interbank Cash Rate Futures WLS)</b>',
        labels={'meeting_date': 'Meeting Date', 'implied_rate': 'Implied Cash Rate (%)'},
        color_discrete_sequence=['#002A54']  # RBA Navy Blue
    )
    fig.update_layout(
        xaxis_title='Meeting Date',
        yaxis_title='Implied Cash Rate (%)',
        template='plotly_white',
        hovermode='x unified',
        height=420
    )
    fig.update_traces(hovertemplate='Meeting Date: %{x|%Y-%m-%d}<br>Implied Rate: %{y:.2f}%<extra></extra>')
    return fig

# ==============================================================================
# 6. PIPELINE EXECUTION & CSV EXPORT
# ==============================================================================
def run_rba_engine():
    prices, as_of = load_rba_futures("data/raw_rates.csv")
    schedule = bootstrap_1m_rba_curve(
        current_rba_rate=CURRENT_RBA_RATE,
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
    curve_output = "data/rba_implied_curve.csv"
    df_curve.to_csv(curve_output, index=False)
    print(f"✓ Saved RBA implied curve to {curve_output}")

    # 2. Export Probabilities CSV
    df_probs = calculate_probabilities(CURRENT_RBA_RATE, schedule)
    df_probs["as_of_date"] = as_of
    probs_output = "data/rba_probabilities.csv"
    df_probs.to_csv(probs_output, index=False)
    print(f"✓ Saved RBA probability distribution to {probs_output}")

    # Console display
    print("\n--- IMPLIED RBA CASH RATE SCHEDULE ---")
    print(df_curve[["meeting_name", "meeting_date", "implied_rate"]].to_string(index=False))

    print("\n--- TARGET PROBABILITY MATRIX (%) ---")
    df_pivot = df_probs.pivot(index="target_rate", columns="meeting_name", values="probability").fillna(0.0)
    print(df_pivot)

    return df_curve, df_probs

if __name__ == "__main__":
    run_rba_engine()