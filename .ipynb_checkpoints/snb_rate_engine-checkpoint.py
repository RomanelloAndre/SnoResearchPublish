# -*- coding: utf-8 -*-
"""
Swiss National Bank (SNB) Implied Policy Rate & Probability Engine
Bootstrapped from Quarterly IMM 3-Month SARON Futures (EUREX / ICE: SARON / FSR3 / J2)
via Regularized Weighted Linear Least Squares (WLS) supporting negative rate bounds.
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
# 1. GLOBAL INPUTS & OFFICIAL SNB ASSESSMENT CALENDAR
# ==============================================================================
CURRENT_SNB_RATE = 0.00   # Current SNB Policy Rate (%)
SPREAD_BPS = 0.04         # SARON fixes ~4 bps below SNB policy rate (+4 bps added to yield)
SMOOTHNESS_LAMBDA = 1e-3  # Regularization penalty to prevent step oscillations

# Standard Quarterly IMM Reference Windows (3rd Wednesday of Mar/Jun/Sep/Dec)
THREE_MONTH_CONTRACTS = {
    "202606": {"start": datetime.date(2026, 6, 17),  "end": datetime.date(2026, 9, 16),  "code": "FSR3M26", "alt": "J2M26"},
    "202609": {"start": datetime.date(2026, 9, 16),  "end": datetime.date(2026, 12, 16), "code": "FSR3U26", "alt": "J2U26"},
    "202612": {"start": datetime.date(2026, 12, 16), "end": datetime.date(2027, 3, 17),  "code": "FSR3Z26", "alt": "J2Z26"},
    "202703": {"start": datetime.date(2027, 3, 17),  "end": datetime.date(2027, 6, 16),  "code": "FSR3H27", "alt": "J2H27"},
    "202706": {"start": datetime.date(2027, 6, 16),  "end": datetime.date(2027, 9, 15),  "code": "FSR3M27", "alt": "J2M27"},
    "202709": {"start": datetime.date(2027, 9, 15),  "end": datetime.date(2027, 12, 15), "code": "FSR3U27", "alt": "J2U27"},
    "202712": {"start": datetime.date(2027, 12, 15), "end": datetime.date(2028, 3, 15),  "code": "FSR3Z27", "alt": "J2Z27"},
    "202803": {"start": datetime.date(2028, 3, 15),  "end": datetime.date(2028, 6, 21),  "code": "FSR3H28", "alt": "J2H28"},
}

# Official SNB Monetary Policy Assessment Dates
# (SNB rate changes take effect on the business day following the assessment)
SNB_MEETINGS = [
    {"name": "SNB Assessment: Dec 10, 2026", "meeting_date": datetime.date(2026, 12, 10), "effective_date": datetime.date(2026, 12, 11)},
    {"name": "SNB Assessment: Mar 18, 2027", "meeting_date": datetime.date(2027, 3, 18), "effective_date": datetime.date(2027, 3, 19)},
    {"name": "SNB Assessment: Jun 17, 2027", "meeting_date": datetime.date(2027, 6, 17), "effective_date": datetime.date(2027, 6, 18)},
    {"name": "SNB Assessment: Sep 23, 2027", "meeting_date": datetime.date(2027, 9, 23), "effective_date": datetime.date(2027, 9, 24)},
    {"name": "SNB Assessment: Dec 16, 2027", "meeting_date": datetime.date(2027, 12, 16), "effective_date": datetime.date(2027, 12, 17)},
    {"name": "SNB Assessment: Mar 16, 2028", "meeting_date": datetime.date(2028, 3, 16), "effective_date": datetime.date(2028, 3, 17)},
]

# ==============================================================================
# 2. AUTOMATED DATA LOADER
# ==============================================================================
def load_snb_futures(csv_path="data/raw_rates.csv"):
    """
    Reads harvested futures quotes from raw_rates.csv and extracts settlement
    prices for Swiss National Bank (SARON) contracts.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found at: {csv_path}. Run fetch_rates.py first.")

    df = pd.read_csv(csv_path)
    df_snb = df[df["central_bank"].str.upper() == "SNB"].copy()

    if df_snb.empty:
        raise ValueError("No records found for SNB in raw_rates.csv.")

    prices = {}
    as_of_date = str(df_snb["date"].iloc[-1])

    for _, row in df_snb.iterrows():
        month_str = str(row["contract_month"]).strip()[:6]
        settle = float(row["settle_price"])
        prices[month_str] = settle

        if "local_symbol" in row and pd.notna(row["local_symbol"]):
            prices[str(row["local_symbol"]).strip()] = settle

    print(f"Loaded {len(df_snb)} SNB futures quotes from {csv_path} (As-of Date: {as_of_date})")
    return prices, as_of_date

# ==============================================================================
# 3. GLOBAL CURVE BOOTSTRAPPER (Regularized WLS Optimization)
# ==============================================================================
def bootstrap_3m_saron_curve(current_snb_rate, spread_bps, prices, as_of_date=None, smoothness_lambda=SMOOTHNESS_LAMBDA, weights=None):
    """
    Solves A * x = b via Bounded Linear Least Squares with smoothing regularization.
    Resolves multi-meeting quarterly overlaps and permits negative interest rates.
    """
    if as_of_date:
        today = datetime.datetime.strptime(as_of_date, "%Y-%m-%d").date()
    else:
        today = datetime.date.today()

    future_meetings = [m for m in SNB_MEETINGS if m["effective_date"] >= today]

    # Match contracts against ingested prices (by month code, localSymbol, or alt ticker)
    valid_contracts = {}
    matched_prices = {}
    for month_key, bounds in THREE_MONTH_CONTRACTS.items():
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

    m1_date = future_meetings[0]["effective_date"]
    active_contracts = {k: v for k, v in valid_contracts.items() if v["end"] > m1_date}

    if not active_contracts:
        print("⚠️ Warning: No active contracts covering post-assessment periods.")
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

        # Baseline days prior to next assessment (priced at current Policy Rate)
        overlap_r0 = max(0, (min(c_end, m1_date) - c_start).days)

        # Fraction of contract days spent in each post-assessment rate regime
        for col_idx in range(n_meetings):
            eff_start = meeting_eff_dates[col_idx]
            eff_end = meeting_eff_dates[col_idx + 1] if (col_idx + 1 < n_meetings) else datetime.date.max

            overlap_start = max(c_start, eff_start)
            overlap_end = min(c_end, eff_end)
            days_at_rate = max(0, (overlap_end - overlap_start).days)
            A[row_idx, col_idx] = days_at_rate / total_days

        b[row_idx] = actual_yield - (overlap_r0 / total_days) * current_snb_rate

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

    # Bound SNB Policy Rate allowing negative rates (-2.00% to +10.00%)
    bounds = (np.full(n_meetings, -2.00), np.full(n_meetings, 10.00))
    res = lsq_linear(A_aug, b_aug, bounds=bounds)

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
def calculate_probabilities(current_snb_rate, implied_schedule):
    """
    Computes step probabilities for 25 bps moves and builds Markov probability trees.
    """
    state_probs = {current_snb_rate: 1.0}
    prev_rate = current_snb_rate
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
    """Generates grouped probability chart across all upcoming SNB assessments."""
    fig = px.bar(
        df_probs,
        x="target_rate",
        y="probability",
        color="meeting_name",
        barmode="group",
        text="probability",
        title="<b>Implied SNB Policy Rate Target Probability Distribution (SARON Futures)</b>",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 105],
        template="plotly_white",
        height=500,
        xaxis_title="SNB Policy Rate Target",
        yaxis_title="Probability (%)",
        legend_title="Assessment Date"
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
        title='<b>Implied SNB Policy Rate Path (Quarterly IMM 3M SARON WLS)</b>',
        labels={'meeting_date': 'Assessment Date', 'implied_rate': 'Implied Policy Rate (%)'},
        color_discrete_sequence=['#D52B1E']  # Swiss Red
    )
    fig.update_layout(
        xaxis_title='SNB Assessment Date',
        yaxis_title='Implied Policy Rate (%)',
        template='plotly_white',
        hovermode='x unified',
        height=420
    )
    fig.update_traces(hovertemplate='Assessment Date: %{x|%Y-%m-%d}<br>Implied Rate: %{y:.2f}%<extra></extra>')
    return fig

# ==============================================================================
# 6. PIPELINE EXECUTION & CSV EXPORT
# ==============================================================================
def run_snb_engine():
    prices, as_of = load_snb_futures("data/raw_rates.csv")
    schedule = bootstrap_3m_saron_curve(
        current_snb_rate=CURRENT_SNB_RATE,
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
    curve_output = "data/snb_implied_curve.csv"
    df_curve.to_csv(curve_output, index=False)
    print(f"✓ Saved SNB implied curve to {curve_output}")

    # 2. Export Probabilities CSV
    df_probs = calculate_probabilities(CURRENT_SNB_RATE, schedule)
    df_probs["as_of_date"] = as_of
    probs_output = "data/snb_probabilities.csv"
    df_probs.to_csv(probs_output, index=False)
    print(f"✓ Saved SNB probability distribution to {probs_output}")

    # Console display
    print("\n--- IMPLIED SNB POLICY RATE SCHEDULE ---")
    print(df_curve[["meeting_name", "meeting_date", "implied_rate"]].to_string(index=False))

    print("\n--- TARGET PROBABILITY MATRIX (%) ---")
    df_pivot = df_probs.pivot(index="target_rate", columns="meeting_name", values="probability").fillna(0.0)
    print(df_pivot)

    return df_curve, df_probs

if __name__ == "__main__":
    run_snb_engine()