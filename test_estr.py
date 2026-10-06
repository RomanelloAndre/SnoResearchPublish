from ib_async import IB, Future, util

util.startLoop()

ib = IB()

try:
    ib.connect('127.0.0.1', 4002, clientId=1)
    ib.reqMarketDataType(3)

    # 1. Test using ST3 (the official Eurex underlying for 3M €STR)
    contract = Future(
        symbol='ST3',
        lastTradeDateOrContractMonth='202612',
        exchange='EUREX',
        currency='EUR'
    )

    qualified = ib.qualifyContracts(contract)

    if qualified:
        print("Contract verified successfully!")
        print("Local Symbol :", contract.localSymbol)
        print("Contract ID  :", contract.conId)
        print("Trading Class:", contract.tradingClass)

        # Pull historical bars
        bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime='',
            durationStr='3 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        df = util.df(bars)
        if not df.empty:
            print("\nRecent Price Data:")
            print(df[['date', 'open', 'high', 'low', 'close', 'volume']])
        else:
            print("\nContract verified, but no trades returned for this far-dated bar.")
    else:
        # Fallback: Query IBKR's database directly for matches
        print("ST3 did not qualify. Searching IBKR system for ESTR contracts...")
        matches = ib.reqMatchingSymbols('ESTR')
        for match in matches[:5]:
            print(f"Match found: Symbol='{match.contract.symbol}', SecType='{match.contract.secType}', Exchange='{match.contract.primaryExchange}'")

finally:
    ib.disconnect()
    print("\nDisconnected.")