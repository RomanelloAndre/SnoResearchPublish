from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from ib_async import IB, Future, util

util.startLoop()

def get_monthly_contracts(num_months=8):
    """Next N consecutive calendar months (Fed Funds, RBA Cash Rate)."""
    now = datetime.now()
    return [(now + relativedelta(months=i)).strftime('%Y%m') for i in range(num_months)]

def get_quarterly_contracts(num_quarters=5):
    """
    Next N IMM quarterly months (March, June, September, December),
    including the currently active quarter that has not yet expired.
    """
    imm_months = [3, 6, 9, 12]
    now = datetime.now()
    quarters = []
    
    # Start 2 months back to capture the active quarterly contract
    current = now - relativedelta(months=2)
    while len(quarters) < num_quarters:
        if current.month in imm_months:
            # IMM contract settlement falls roughly 3 months after start
            expiry_approx = current + relativedelta(months=3)
            if expiry_approx >= now:
                quarters.append(current.strftime('%Y%m'))
        current += relativedelta(months=1)
        
    return quarters

def get_contract_bar(ib, contract):
    """Fetches daily settlement bar using TRADES with MIDPOINT fallback."""
    # Attempt 1: TRADES
    try:
        bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            timeout=8
        )
        if bars:
            return bars[-1].close
    except Exception:
        pass

    # Attempt 2: MIDPOINT Fallback (handles zero-volume deferred contracts)
    try:
        bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 day',
            whatToShow='MIDPOINT',
            useRTH=True,
            timeout=8
        )
        if bars:
            return bars[-1].close
    except Exception:
        pass

    return None

def harvest_rates():
    ib = IB()
    print("Connecting to IB Gateway on 127.0.0.1:4002 (Paper Trading)...")
    ib.connect('127.0.0.1', 4002, clientId=10)

    # Free delayed data tier ($0)
    ib.reqMarketDataType(3)

    # Complete 7 Central Bank Universe
    instruments = [
        # 1. US Federal Reserve (CBOT 30-Day Fed Funds)
        {
            'central_bank': 'FED',
            'symbol': 'ZQ',
            'exchange': 'CBOT',
            'currency': 'USD',
            'months': get_monthly_contracts(8)
        },
        # 2. European Central Bank (Eurex 3-Month €STR)
        {
            'central_bank': 'ECB',
            'symbol': 'ST3',
            'exchange': 'EUREX',
            'currency': 'EUR',
            'months': get_quarterly_contracts(5)
        },
        # 3. Bank of England (ICE Europe 3-Month SONIA)
        {
            'central_bank': 'BOE',
            'symbol': 'SONIA.N',
            'exchange': 'ICEEU',
            'currency': 'GBP',
            'months': get_quarterly_contracts(5)
        },
        # 4. Bank of Japan (OSE 3-Month TONA)
        {
            'central_bank': 'BOJ',
            'symbol': 'TONA',
            'exchange': 'OSE.JPN',
            'currency': 'JPY',
            'months': get_quarterly_contracts(5)
        },
        # 5. Reserve Bank of Australia (ASX/SNFE 30-Day Cash Rate)
        {
            'central_bank': 'RBA',
            'symbol': 'IB',
            'exchange': 'SNFE',
            'currency': 'AUD',
            'months': get_monthly_contracts(8)
        },
        # 6. Bank of Canada (Bourse de Montreal 3-Month CORRA)
        {
            'central_bank': 'BOC',
            'symbol': 'CRA',
            'exchange': 'CDE',
            'currency': 'CAD',
            'months': get_quarterly_contracts(5)
        },
        # 7. Swiss National Bank (Eurex 3-Month SARON)
        {
            'central_bank': 'SNB',
            'symbol': 'SARON',
            'exchange': 'EUREX',
            'currency': 'CHF',
            'months': get_quarterly_contracts(5)
        }
    ]

    records = []

    try:
        for inst in instruments:
            cb = inst['central_bank']
            print(f"\nFetching {cb} ({inst['symbol']} on {inst['exchange']})...")
            
            for month in inst['months']:
                contract = Future(
                    symbol=inst['symbol'],
                    lastTradeDateOrContractMonth=month,
                    exchange=inst['exchange'],
                    currency=inst['currency']
                )

                qualified = ib.qualifyContracts(contract)
                if not qualified or not getattr(contract, 'conId', None):
                    print(f"  ⚠️ Could not qualify: {inst['symbol']} {month}")
                    continue

                # Normalizes ICE and Eurex timestamp strings like '20270316 18:00:00 GB' back to '20270316'
                if contract.lastTradeDateOrContractMonth and len(contract.lastTradeDateOrContractMonth) > 8:
                    contract.lastTradeDateOrContractMonth = contract.lastTradeDateOrContractMonth[:8]

                close_price = get_contract_bar(ib, contract)

                if close_price is not None:
                    implied_rate = round(100.0 - close_price, 4)
                    records.append({
                        'central_bank': cb,
                        'contract_month': month,
                        'local_symbol': contract.localSymbol,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'settle_price': round(float(close_price), 4),
                        'implied_rate': implied_rate
                    })
                    print(f"  ✓ {contract.localSymbol}: {close_price:.4f} (Implied: {implied_rate}%)")
                else:
                    print(f"  ⚠️ No price returned for {contract.localSymbol}")
                
                ib.sleep(0.3)

    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway.")

    df = pd.DataFrame(records)
    output_path = 'data/raw_rates.csv'
    
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\n🎉 Successfully saved {len(df)} contract quotes to {output_path}")
    else:
        print("\n⚠️ No contract quotes were collected.")

    return df

if __name__ == '__main__':
    df_rates = harvest_rates()
    print(df_rates)