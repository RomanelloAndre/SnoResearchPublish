import datetime
import math
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ==============================================================================
# 1. ECB CONFIGURATION & GOVERNING COUNCIL CALENDAR
# ==============================================================================
CURRENT_DFR = 2.50    # Current ECB Deposit Facility Rate (%)
SPREAD_BPS = 0.07     # €STR fixes ~7 bps below DFR

# Official ECB Meetings & Reserve Maintenance Period Effective Dates
# (Decisions take effect on the Wednesday following the announcement)
ECB_MEETINGS = [
    {"name": "ECB: Sep 10, 2026", "meeting_date": datetime.date(2026, 9, 10), "effective_date": datetime.date(2026, 9, 16)},
    {"name": "ECB: Oct 29, 2026", "meeting_date": datetime.date(2026, 10, 29), "effective_date": datetime.date(2026, 11, 4)},
    {"name": "ECB: Dec 17, 2026", "meeting_date": datetime.date(2026, 12, 17), "effective_date": datetime.date(2026, 12, 23)},
    {"name": "ECB: Feb 04, 2027", "meeting_date": datetime.date(2027, 2, 4),   "effective_date": datetime.date(2027, 2, 10)},
    {"name": "ECB: Mar 18, 2027", "meeting_date": datetime.date(2027, 3, 18),  "effective_date": datetime.date(2027, 3, 24)},
    {"name": "ECB: Apr 29, 2027", "meeting_date": datetime.date(2027, 4, 29),  "effective_date": datetime.date(2027, 5, 5)},
    {"name": "ECB: Jun 10, 2027", "meeting_date": datetime.date(2027, 6, 10),  "effective_date": datetime.date(2027, 6, 16)},
    {"name": "ECB: Jul 22, 2027", "meeting_date": datetime.date(2027, 7, 22),  "effective_date": datetime.date(2027, 7, 28)},
    {"name": "ECB: Sep 09, 2027", "meeting_date": datetime.date(2027, 9, 9),   "effective_date": datetime.date(2027, 9, 15)},
    {"name": "ECB: Oct 28, 2027", "meeting_date": datetime.date(2027, 10, 28), "effective_date": datetime.date(2027, 11, 3)},
    {"name": "ECB: Dec 16, 2027", "meeting_date": datetime.date(2027, 12, 16), "effective_date": datetime.date(2027, 12, 22)},
]

# ==============================================================================
# 2. HELPER FUNCTIONS FOR IMM EXPIRIES & SETTLEMENT PARSING
# ==============================================================================
def get_third_wednesday(year: int, month: int) -> datetime.date:
    """Calculates the 3rd Wednesday of a given month (Quarterly IMM delivery day)."""
    d = datetime.date(year, month, 1)
    first_wed = 1 + (2 - d.weekday()) % 7
    return datetime.date(year, month, first_wed + 14)

def get_imm_bounds(month_str: str):
    """
    Given 'YYYYMM' (e.g. '202612'), returns the start and end of the 3-month accrual window.
    """
    yr = int(month_str[:4])
    mo = int(month_str[4:])
    start_date = get_third_wednesday(yr, mo)
    
    end_mo = mo + 3
    end_yr = yr
    if end_mo > 12:
        end_mo -= 12
        end_yr += 1
    end_date = get_third_wednesday(end_yr, end_mo)
    return start_date, end_date

def load_ecb_prices_from_csv(csv_path="data/raw_rates.csv"):
    df = pd.read_csv(csv_path)
    ecb_df = df[df["central_bank"] == "ECB"]
    
    prices = {}
    for _, row in ecb_df.iterrows():
        m_str = str(row["contract_month"])
        prices[m_str] = float(row["settle_price"])
    return prices

# ==============================================================================
# 3. REGULARIZED WLS DEAVERAGING ENGINE
# ==============================================================================
def deaverage_ecb_curve(current_dfr, spread_bps, prices, smoothness_lambda=1e-3):
    today = datetime.date.today()
    future_meetings = [m for m in ECB_MEETINGS if m["effective_date"] > today]
    
    if not future_meetings or not prices:
        return []

    # Map bounds for all collected contracts
    contract_bounds = {}
    for m_str, price in prices.items():
        c_start, c_end = get_imm_bounds(m_str)
        contract_bounds[m_str] = {
            "start": c_start,
            "end": c_end,
            "days": (c_end - c_start).days,
            "price": price
        }

    first_eff_date = future_meetings[0]["effective_date"]
    active_contracts = {k: v for k, v in contract_bounds.items() if v["end"] > first_eff_date}

    if not active_contracts:
        return []

    max_date = max(c["end"] for c in active_contracts.values())
    relevant_meetings = [m for m in future_meetings if m["effective_date"] < max_date]

    n_contracts = len(active_contracts)
    n_meetings = len(relevant_meetings)

    if n_meetings == 0 or n_contracts == 0:
        return []

    A = np.zeros((n_contracts, n_meetings))
    b = np.zeros(n_contracts)
    meeting_eff_dates = [m["effective_date"] for m in relevant_meetings]

    for row_idx, (code, bounds) in enumerate(active_contracts.items()):
        c_start = bounds["start"]
        c_end = bounds["end"]
        total_days = bounds["days"]

        # Compounding convexity adjustment
        raw_yield = (100.0 - bounds["price"]) + spread_bps
        compounding_bias = 0.5 * ((raw_yield / 100.0) ** 2) * (total_days / 360.0) * 100.0
        adjusted_yield = raw_yield - compounding_bias

        # Accrual prior to the first upcoming meeting
        m1_date = meeting_eff_dates[0]
        overlap_r0 = max(0, (min(c_end, m1_date) - c_start).days)

        for col_idx in range(n_meetings):
            eff_start = meeting_eff_dates[col_idx]
            eff_end = meeting_eff_dates[col_idx + 1] if (col_idx + 1 < n_meetings) else datetime.date.max

            overlap_start = max(c_start, eff_start)
            overlap_end = min(c_end, eff_end)
            days_at_rate = max(0, (overlap_end - overlap_start).days)
            A[row_idx, col_idx] = days_at_rate / total_days

        b[row_idx] = adjusted_yield - (overlap_r0 / total_days) * current_dfr

    # Policy inertia penalty: minimizes (r_{t+1} - r_t)^2
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

    res = lsq_linear(A_aug, b_aug, bounds=(np.zeros(n_meetings), np.full(n_meetings, 10.0)))

    schedule = []
    for i, m in enumerate(relevant_meetings):
        schedule.append({
            "meeting_name": m["name"],
            "meeting_date": str(m["meeting_date"]),
            "effective_date": str(m["effective_date"]),
            "implied_post_rate": round(float(res.x[i]), 3)
        })

    return schedule

# ==============================================================================
# 4. MARKOV CHAIN PROBABILITY TREE
# ==============================================================================
def calculate_ecb_probabilities(current_dfr, schedule):
    state_probs = {current_dfr: 1.0}
    prev_rate = current_dfr
    records = []

    for mtg in schedule:
        mtg_name = mtg["meeting_name"]
        curr_implied_rate = mtg["implied_post_rate"]

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
            if prob_pct > 0.01:
                records.append({
                    "meeting_name": mtg_name,
                    "target_rate": f"{rate_val:.2f}%",
                    "rate_num": rate_val,
                    "probability": prob_pct
                })

        state_probs = next_state_probs
        prev_rate = curr_implied_rate

    return pd.DataFrame(records)

# ==============================================================================
# 5. EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("Loading ECB contracts from data/raw_rates.csv...")
    prices = load_ecb_prices_from_csv("data/raw_rates.csv")
    print(f"Loaded {len(prices)} contracts: {list(prices.keys())}")

    schedule = deaverage_ecb_curve(CURRENT_DFR, SPREAD_BPS, prices)

    if schedule:
        df_curve = pd.DataFrame(schedule)
        df_curve.to_csv("data/ecb_curve.csv", index=False)
        print("\nSaved Implied ECB Curve to 'data/ecb_curve.csv':")
        print(df_curve[["meeting_name", "meeting_date", "implied_post_rate"]])

        df_probs = calculate_ecb_probabilities(CURRENT_DFR, schedule)
        df_probs.to_csv("data/ecb_probabilities.csv", index=False)
        print("\nSaved ECB Probabilities to 'data/ecb_probabilities.csv':")
        print(df_probs.head(8))
    else:
        print("Could not match contracts to upcoming ECB meeting windows.")