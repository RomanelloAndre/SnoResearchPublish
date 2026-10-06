import calendar
import datetime
import math
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ==============================================================================
# 1. CONFIGURATION & OFFICIAL FOMC CALENDAR
# ==============================================================================
CURRENT_EFFR = 3.63  # Update if EFFR changes

FOMC_MEETINGS = [
    # 2026
    {"name": "FOMC: Jan 28, 2026", "meeting_date": datetime.date(2026, 1, 28)},
    {"name": "FOMC: Mar 18, 2026", "meeting_date": datetime.date(2026, 3, 18)},
    {"name": "FOMC: Apr 29, 2026", "meeting_date": datetime.date(2026, 4, 29)},
    {"name": "FOMC: Jun 17, 2026", "meeting_date": datetime.date(2026, 6, 17)},
    {"name": "FOMC: Jul 29, 2026", "meeting_date": datetime.date(2026, 7, 29)},
    {"name": "FOMC: Sep 16, 2026", "meeting_date": datetime.date(2026, 9, 16)},
    {"name": "FOMC: Oct 28, 2026", "meeting_date": datetime.date(2026, 10, 28)},
    {"name": "FOMC: Dec 09, 2026", "meeting_date": datetime.date(2026, 12, 9)},
    # 2027
    {"name": "FOMC: Jan 27, 2027", "meeting_date": datetime.date(2027, 1, 27)},
    {"name": "FOMC: Mar 17, 2027", "meeting_date": datetime.date(2027, 3, 17)},
    {"name": "FOMC: Apr 28, 2027", "meeting_date": datetime.date(2027, 4, 28)},
    {"name": "FOMC: Jun 09, 2027", "meeting_date": datetime.date(2027, 6, 9)},
    {"name": "FOMC: Jul 28, 2027", "meeting_date": datetime.date(2027, 7, 28)},
    {"name": "FOMC: Sep 15, 2027", "meeting_date": datetime.date(2027, 9, 15)},
    {"name": "FOMC: Oct 27, 2027", "meeting_date": datetime.date(2027, 10, 27)},
    {"name": "FOMC: Dec 08, 2027", "meeting_date": datetime.date(2027, 12, 8)},
]

# ==============================================================================
# 2. LOAD LOCAL IBKR SETTLEMENT DATA
# ==============================================================================
def load_fed_prices_from_csv(csv_path="data/raw_rates.csv"):
    df = pd.read_csv(csv_path)
    fed_df = df[df["central_bank"] == "FED"]
    
    monthly_prices = {}
    for _, row in fed_df.iterrows():
        month_str = str(row["contract_month"])
        yr = int(month_str[:4])
        mo = int(month_str[4:])
        monthly_prices[(yr, mo)] = float(row["settle_price"])
        
    return monthly_prices

# ==============================================================================
# 3. WLS CURVE SOLVER
# ==============================================================================
def bootstrap_fed_curve_wls(current_effr, monthly_prices, meetings, smoothness_lambda=1e-3):
    today = datetime.date.today()

    future_meetings = []
    for m in meetings:
        eff_date = m["meeting_date"] + datetime.timedelta(days=1)
        if eff_date >= today:
            future_meetings.append({
                "name": m["name"],
                "meeting_date": m["meeting_date"],
                "effective_date": eff_date
            })

    if not future_meetings or not monthly_prices:
        return []

    contract_bounds = {}
    for (yr, mo), price in monthly_prices.items():
        start_date = datetime.date(yr, mo, 1)
        days_in_month = calendar.monthrange(yr, mo)[1]
        end_date = start_date + datetime.timedelta(days=days_in_month)
        contract_bounds[(yr, mo)] = {
            "start": start_date,
            "end": end_date,
            "days": days_in_month,
            "price": price
        }

    m1_date = future_meetings[0]["effective_date"]
    active_contracts = {k: v for k, v in contract_bounds.items() if v["end"] > m1_date}
    if not active_contracts:
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

    for row_idx, ((yr, mo), bounds) in enumerate(active_contracts.items()):
        c_start = bounds["start"]
        c_end = bounds["end"]
        total_days = bounds["days"]
        actual_yield = 100.0 - bounds["price"]

        overlap_r0 = max(0, (min(c_end, m1_date) - c_start).days)

        for col_idx in range(n_meetings):
            eff_start = meeting_eff_dates[col_idx]
            eff_end = meeting_eff_dates[col_idx + 1] if (col_idx + 1 < n_meetings) else datetime.date.max

            overlap_start = max(c_start, eff_start)
            overlap_end = min(c_end, eff_end)
            days_at_rate = max(0, (overlap_end - overlap_start).days)
            A[row_idx, col_idx] = days_at_rate / total_days

        b[row_idx] = actual_yield - (overlap_r0 / total_days) * current_effr

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

    bounds = (np.zeros(n_meetings), np.full(n_meetings, 15.00))
    res = lsq_linear(A_aug, b_aug, bounds=bounds)

    schedule = []
    for i, m in enumerate(relevant_meetings):
        schedule.append({
            "meeting_name": m["name"],
            "meeting_date": str(m["meeting_date"]),
            "implied_post_rate": round(float(res.x[i]), 3)
        })

    return schedule

# ==============================================================================
# 4. PROBABILITY CALCULATION
# ==============================================================================
def calculate_probabilities(current_effr, schedule):
    state_probs = {current_effr: 1.0}
    prev_rate = current_effr
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
                    "target_bracket": f"{rate_val - 0.125:.2f}% - {rate_val + 0.125:.2f}%",
                    "rate_mid": rate_val,
                    "probability": prob_pct
                })

        state_probs = next_state_probs
        prev_rate = curr_implied_rate

    return pd.DataFrame(records)

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Loading data from data/raw_rates.csv...")
    prices = load_fed_prices_from_csv("data/raw_rates.csv")
    print(f"Loaded {len(prices)} Fed Funds contract months: {list(prices.keys())}")

    schedule = bootstrap_fed_curve_wls(CURRENT_EFFR, prices, FOMC_MEETINGS)

    if schedule:
        df_curve = pd.DataFrame(schedule)
        df_curve.to_csv("data/fomc_curve.csv", index=False)
        print("\nSaved Implied Path to 'data/fomc_curve.csv':")
        print(df_curve)

        df_probs = calculate_probabilities(CURRENT_EFFR, schedule)
        df_probs.to_csv("data/fomc_probabilities.csv", index=False)
        print("\nSaved Meeting Probabilities to 'data/fomc_probabilities.csv':")
        print(df_probs.head(10))
    else:
        print("No meetings matched the contract window.")