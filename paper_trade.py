import argparse
import json
import os
import numpy as np
import pandas as pd

from data_fetcher import update_csv
from pattern_backtest import load_ohlc, collect_events, DEFAULTS
from smart_money import collect_smc_events, ORDER
from strategy_lab import FILTERS, add_confluence
from trade_sim import atr, simulate_trade_live, DEFAULT_COST

# Paper trading: nothing is ever sent to an exchange.
#
# Each run downloads the newest daily candles, finds pattern signals that appeared since the paper-trading
# start date, and works out what each trade would have done with the rules in paper_strategies.json
# (next-open entry, ATR stop/target, max hold, 0.15% cost per side). Results go to paper_trades.csv.
# The simulation is deterministic, so running it every day just extends the log; no state is lost.
#
#   python paper_trade.py            # update data, print open trades and the running record
#   python paper_trade.py --offline  # skip the download

COINS = ['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']
STRATEGY_FILE = 'paper_strategies.json'
STATE_FILE = 'paper_state.json'
LOG_FILE = 'paper_trades.csv'


def load_strategies(path: str = STRATEGY_FILE) -> list:
    with open(path) as f:
        return json.load(f)['strategies']


def get_start_date(override: str = None) -> pd.Timestamp:
    # The date paper trading began. Signals before it are ignored. Stored on first run.
    if override:
        start = override
    elif os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return pd.Timestamp(json.load(f)['start'])
    else:
        start = str(pd.Timestamp.today().date())
    with open(STATE_FILE, 'w') as f:
        json.dump({'start': start}, f)
    return pd.Timestamp(start)


def build_coin(symbol: str, data_dir: str, download: bool) -> dict:
    path = f'{data_dir}/{symbol}86400.csv'
    if download:
        update_csv(path, symbol, '1d')
    ohlc = load_ohlc(path)
    cfg = DEFAULTS['daily']
    v = pd.read_csv(path, usecols=['date', 'volume'], parse_dates=['date']).set_index('date')['volume']
    feat = pd.DataFrame({'close': ohlc['close']})
    feat['sma50'] = feat['close'].rolling(50).mean()
    feat['sma200'] = feat['close'].rolling(200).mean()
    feat['vol_ratio'] = (v / v.rolling(20).mean()).reindex(ohlc.index)
    events = pd.concat([collect_events(ohlc, cfg['order'], cfg['tri_lookback']),
                        collect_smc_events(ohlc, ORDER['daily'])], ignore_index=True)
    return dict(ohlc=ohlc, events=events, feat=feat, atr=atr(ohlc))


def run(strategies: list, start: pd.Timestamp, data_dir: str = 'data', download: bool = True,
        cost: float = DEFAULT_COST) -> pd.DataFrame:
    coins = {s: build_coin(s, data_dir, download) for s in COINS}
    for st in strategies:   # combined signals, if any strategy uses them
        if st['pattern'] == 'macd_and_rsi_bear':
            add_confluence(coins, 'macd_and_rsi_bear', 'macd_bear_div', 'rsi_bear_div', -1, 3)
        if st['pattern'] == 'macd_and_rsi_bull':
            add_confluence(coins, 'macd_and_rsi_bull', 'macd_bull_div', 'rsi_bull_div', 1, 3)

    rows = []
    for st in strategies:
        d = int(st['direction'])
        for sym, cd in coins.items():
            ohlc, ev = cd['ohlc'], cd['events']
            o, h, l, c = (ohlc[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
            ok = np.nan_to_num(FILTERS[st.get('filter', 'none')](cd['feat'], d), nan=0).astype(bool)
            sig = np.sort(ev.loc[(ev['pattern'] == st['pattern']) & (ev['direction'] == d), 'i'].to_numpy())
            busy_until = -1
            for i in sig:
                if ohlc.index[i] < start or not ok[i] or i <= busy_until:
                    continue
                r = simulate_trade_live(o, h, l, c, cd['atr'], int(i), d, st.get('stop'), st.get('tgt'),
                                        int(st['hold']), cost)
                if r is None:
                    continue
                busy_until = r.get('exit_i', int(i) + int(st['hold'])) if r['status'] != 'PENDING' else len(c)
                row = dict(strategy=st['name'], coin=sym, side='long' if d > 0 else 'short',
                           signal_date=ohlc.index[i].date(), status=r['status'])
                if r['status'] == 'PENDING':
                    row.update(entry_date=ohlc.index[-1].date() + pd.Timedelta(days=1))
                else:
                    row.update(entry_date=ohlc.index[r['entry_i']].date(), entry_price=r['entry'],
                               stop=r['stop'], target=r['tgt'], last_price=r['price'],
                               exit_date=ohlc.index[r['exit_i']].date() if r['status'] == 'CLOSED' else None,
                               reason=r['reason'], net_pct=100 * r['net'])
                rows.append(row)
    return pd.DataFrame(rows)


KEY = ['strategy', 'coin', 'signal_date']


def what_changed(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    # Rows that are new since the previous run, or whose status changed (e.g. PENDING -> OPEN -> CLOSED).
    if old is None or old.empty:
        return new.copy()
    m = new.merge(old[KEY + ['status']], on=KEY, how='left', suffixes=('', '_old'))
    return m[m['status_old'].isna() | (m['status'] != m['status_old'])].drop(columns='status_old')


def report(log: pd.DataFrame, start: pd.Timestamp, changed: pd.DataFrame = None):
    pd.set_option('display.width', 220)
    print(f'Paper trading since {start.date()}  (no real orders are placed)\n')
    if changed is not None and len(changed):
        print('NEW OR UPDATED SINCE LAST RUN')
        cols = [c for c in ['strategy', 'coin', 'side', 'signal_date', 'status', 'reason', 'entry_price', 'net_pct'] if c in changed.columns]
        print(changed[cols].round(3).to_string(index=False), '\n')
    elif changed is not None:
        print('Nothing new since the last run.\n')
    if log.empty:
        print('No signals yet. Run again after new daily candles close.')
        return
    for status, title in (('PENDING', 'Signals waiting for tomorrow\'s open'), ('OPEN', 'Open trades (marked at last close)')):
        part = log[log['status'] == status]
        if len(part):
            print(title)
            cols = ['strategy', 'coin', 'side', 'signal_date', 'entry_date'] + \
                   (['entry_price', 'stop', 'target', 'last_price', 'net_pct'] if status == 'OPEN' else [])
            print(part[cols].round(4).to_string(index=False), '\n')
    closed = log[log['status'] == 'CLOSED']
    if len(closed):
        print(f'Closed trades: {len(closed)}   win rate {(closed["net_pct"] > 0).mean():.0%}   '
              f'average net {closed["net_pct"].mean():.2f}%   total net {closed["net_pct"].sum():.1f}%')
        print(closed.groupby('strategy')['net_pct'].agg(['count', 'mean', 'sum']).round(2).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper-trade the configured pattern strategies.')
    parser.add_argument('--offline', action='store_true', help='Do not download new candles')
    parser.add_argument('--start', default=None, help='Paper trading start date (default: first run date)')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    start = get_start_date(args.start)
    old = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 2 else None
    log = run(load_strategies(), start, args.data_dir, download=not args.offline)
    if old is not None:
        old['signal_date'] = old['signal_date'].astype(str)
    log['signal_date'] = log['signal_date'].astype(str) if len(log) else log.get('signal_date')
    changed = what_changed(old, log) if len(log) else pd.DataFrame()
    log.to_csv(LOG_FILE, index=False)
    report(log, start, changed)
