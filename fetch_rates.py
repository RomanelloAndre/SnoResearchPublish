from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from ib_async import IB, Future, util

# Enable event loop for Jupyter / standard execution
util.startLoop()

def get_monthly_contracts(num_months=8):
    """Next N consecutive calendar months (e.g. for Fed Funds)."""
    now = datetime.now()
    return [(now + relativedelta(months=i)).strftime('%Y%m') for i in range(num_months)]

def get_quarterly_contracts(num_quarters=4):
    """Next N IMM quarterly months (March, June, September, December)."""
    imm_months = [3, 6, 9, 12]
    now = datetime.now()
    quarters = []
    
    current = now
    while len(quarters) < num_quarters:
        if current.month in imm_months and current >= now:
            quarters.append(current.strftime('%Y%m'))
        current += relativedelta(months=1)
        
    return quarters

def harvest_rates():
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=1)
    ib.reqMarketDataType(3)  # Delayed / free settlement data

    # Central bank definitions to pull
    instruments = [
        {
            'central_bank': 'FED',
            'symbol': 'ZQ',
            'exchange': 'CBOT',
            'currency': 'USD',
            'months': get_monthly_contracts(8)
        },
        {
            'central_bank': 'ECB',
            'symbol': 'ST3',
            'exchange': 'EUREX',
            'currency': 'EUR',
            'months': get_quarterly_contracts(4)
        }
    ]

    records = []

    try:
        for inst in instruments:
            print(f"\nFetching {inst['central_bank']} ({inst['symbol']})...")
            
            for month in inst['months']:
                contract = Future(
                    symbol=inst['symbol'],
                    lastTradeDateOrContractMonth=month,
                    exchange=inst['exchange'],
                    currency=inst['currency']
                )

                if not ib.qualifyContracts(contract):
                    print(f"  Could not qualify: {inst['symbol']} {month}")
                    continue

                bars = ib.reqHistoricalData(
                    contract=contract,
                    endDateTime='',
                    durationStr='2 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True
                )

                if bars:
                    latest = bars[-1]
                    records.append({
                        'central_bank': inst['central_bank'],
                        'contract_month': month,
                        'local_symbol': contract.localSymbol,
                        'date': str(latest.date),
                        'settle_price': latest.close,
                        'implied_rate': round(100.0 - latest.close, 4)
                    })
                    print(f"  {contract.localSymbol}: {latest.close} (Implied Rate: {round(100.0 - latest.close, 4)}%)")
                
                # Small pause to respect IBKR rate pacing
                ib.sleep(0.4)

    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway.")

    # Convert to DataFrame and save to disk
    df = pd.DataFrame(records)
    output_path = 'data/raw_rates.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} contract quotes to {output_path}")
    return df

if __name__ == '__main__':
    df_rates = harvest_rates()
    print(df_rates)