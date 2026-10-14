import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

# Public Binance kline endpoints (no API key needed). The second is a
# market-data-only mirror that works in some regions where the first is blocked.
BINANCE_HOSTS = [
    'https://api.binance.com',
    'https://data-api.binance.vision',
]

INTERVAL_SECONDS = {
    '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800,
}


def _get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TechnicalAnalysisAutomation'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_klines(symbol: str, interval: str, start: str, end: str = None) -> pd.DataFrame:
    # Download OHLCV candles from Binance between start and end (UTC dates/times).
    start_ms = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz='UTC').timestamp() * 1000) if end else int(time.time() * 1000)

    last_err = None
    for host in BINANCE_HOSTS:
        rows = []
        cursor = start_ms
        try:
            while cursor < end_ms:
                params = urllib.parse.urlencode({
                    'symbol': symbol, 'interval': interval,
                    'startTime': cursor, 'endTime': end_ms, 'limit': 1000,
                })
                batch = _get_json(f'{host}/api/v3/klines?{params}')
                if not batch:
                    break
                rows.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(0.2) # Be polite to the rate limiter
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f'Could not download klines from any Binance host: {last_err}')

    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Drop the candle that is still forming
    now_ms = int(time.time() * 1000)
    df = df[df['date'] + pd.Timedelta(seconds=INTERVAL_SECONDS[interval]) <= pd.Timestamp(now_ms, unit='ms')]
    return df.drop_duplicates('date').reset_index(drop=True)


def update_csv(path: str, symbol: str, interval: str, start: str = '2018-01-01') -> pd.DataFrame:
    # Append new candles to an existing CSV (same format as BTCUSDT86400.csv), or create it.
    if os.path.exists(path):
        old = pd.read_csv(path)
        old['date'] = pd.to_datetime(old['date'])
        start = str(old['date'].iloc[-1] + pd.Timedelta(seconds=INTERVAL_SECONDS[interval]))
    else:
        old = None

    new = fetch_klines(symbol, interval, start)
    if old is not None:
        if 'volume' not in old.columns:
            new = new.drop(columns='volume')
        df = pd.concat([old, new]).drop_duplicates('date', keep='last')
    else:
        df = new

    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download recent OHLC candles from Binance.')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='1d', choices=list(INTERVAL_SECONDS))
    parser.add_argument('--start', default='2018-01-01', help='Start date when creating a new file')
    parser.add_argument('--out', default=None,
                        help='CSV path (default: <SYMBOL><seconds>.csv, e.g. BTCUSDT86400.csv). '
                             'Existing files are extended with the newest candles.')
    args = parser.parse_args()

    out = args.out or f'{args.symbol}{INTERVAL_SECONDS[args.interval]}.csv'
    df = update_csv(out, args.symbol, args.interval, args.start)
    print(f'{out}: {len(df)} candles, {df["date"].iloc[0]} -> {df["date"].iloc[-1]}')
