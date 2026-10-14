import itertools
import numpy as np
import pandas as pd

from cross_coin import COINS, prepare, benjamini_hochberg
from pattern_backtest import load_ohlc, collect_events, DEFAULTS
from trade_sim import (run_trades, trade_stats, random_entry_baseline, DEFAULT_COST, atr)

# Strict "does it hold up?" protocol for trading strategies.
#
#  1. A strategy = pattern + entry filter + exit rules (stop / target / max hold).
#  2. The filter and exit rules are chosen using trades that ENTER BEFORE test_start only.
#  3. The chosen strategy is run unchanged on test_start onward.
#  4. It is compared with random entries that obey the SAME filter, direction and exit rules,
#     so a filter or a market drift can't create an edge by itself.
#  5. P-values are corrected for the number of strategies tried (Benjamini-Hochberg).
#
# "Proven" (decided before looking at results) requires ALL of:
#     >= 30 trades in the recent period, q-value < 0.10 against filter-matched random entries,
#     positive average net return from 2025 on, and a positive average on >= 4 of the 6 coins.

# ---------------------------------------------------------------- data

def prepare_4h(symbols, test_start: str, data_dir: str = 'data') -> dict:
    # 4-hour candles resampled from the hourly files, then every detector is run on them.
    test_start = pd.Timestamp(test_start)
    cfg = dict(order=4, tri_lookback=40)
    out = {}
    for sym in symbols:
        h = load_ohlc(f'{data_dir}/{sym}3600.csv')
        o = h.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        out[sym] = dict(ohlc=o, events=collect_events(o, cfg['order'], cfg['tri_lookback']),
                        recent=np.asarray(o.index >= test_start))
    return out


def add_features(coins: dict, data_dir: str = 'data', tf_secs: int = 86400):
    # Moving averages and volume ratio at every bar (each uses only data up to that bar).
    for sym, cd in coins.items():
        o = cd['ohlc']
        c = o['close']
        feat = pd.DataFrame(index=o.index)
        feat['close'] = c
        feat['sma50'] = c.rolling(50).mean()
        feat['sma200'] = c.rolling(200).mean()
        vol = pd.read_csv(f'{data_dir}/{sym}{tf_secs}.csv', usecols=['date', 'volume'], parse_dates=['date']) \
            if tf_secs == 86400 else None
        if vol is not None:
            v = vol.set_index('date')['volume'].reindex(o.index)
            feat['vol_ratio'] = v / v.rolling(20).mean()
        else:
            feat['vol_ratio'] = np.nan
        cd['feat'] = feat
        cd['atr'] = atr(o)


# ---------------------------------------------------------------- filters

def _with_trend(col):
    # With the trade: shorts when price is below the average, longs when above.
    return lambda f, d: ((f['close'] < f[col]) if d < 0 else (f['close'] > f[col])).to_numpy()


def _against_trend(col):
    return lambda f, d: ((f['close'] > f[col]) if d < 0 else (f['close'] < f[col])).to_numpy()


FILTERS = {
    'none': lambda f, d: np.ones(len(f), dtype=bool),
    'with_sma50': _with_trend('sma50'),
    'against_sma50': _against_trend('sma50'),
    'with_sma200': _with_trend('sma200'),
    'against_sma200': _against_trend('sma200'),
    'vol_spike': lambda f, d: (f['vol_ratio'] > 1.5).to_numpy(),
    'vol_quiet': lambda f, d: (f['vol_ratio'] < 0.8).to_numpy(),
}


def eligible_masks(coins: dict, filt: str, direction: int) -> dict:
    return {sym: np.nan_to_num(FILTERS[filt](cd['feat'], direction), nan=0).astype(bool)
            for sym, cd in coins.items() if 'feat' in cd}


# ---------------------------------------------------------------- confluence

def add_confluence(coins: dict, name: str, pat_a: str, pat_b: str, direction: int, window: int):
    # New pattern `name`: fires on the later of pat_a / pat_b when the other fired within `window` bars.
    for cd in coins.values():
        ev = cd['events']
        a = ev.loc[(ev['pattern'] == pat_a) & (ev['direction'] == direction), 'i'].to_numpy()
        b = ev.loc[(ev['pattern'] == pat_b) & (ev['direction'] == direction), 'i'].to_numpy()
        idx = sorted({max(x, y) for x in a for y in b if abs(x - y) <= window})
        if idx:
            new = pd.DataFrame({'i': idx, 'pattern': name, 'direction': direction})
            new['date'] = cd['ohlc'].index[new['i']]
            cd['events'] = pd.concat([cd['events'], new], ignore_index=True)


# ---------------------------------------------------------------- proof protocol

GRID_DAILY = dict(stop=[None, 1.0, 2.0], tgt=[None, 2.0, 3.0], hold=[5, 10, 20])
GRID_4H = dict(stop=[None, 1.0, 2.0], tgt=[None, 2.0, 3.0], hold=[12, 30, 60])
nn = lambda x: None if pd.isna(x) else float(x)


def tune(coins, pattern, direction, test_start, grid, filters, cost=DEFAULT_COST, min_trades=25):
    test_start = pd.Timestamp(test_start)
    masks = {f: eligible_masks(coins, f, direction) for f in filters}
    best = None
    for f in filters:
        for stop, tgt, hold in itertools.product(grid['stop'], grid['tgt'], grid['hold']):
            t = run_trades(coins, pattern, direction, stop, tgt, hold, cost, eligible=masks[f])
            t = t[t['entry'] < test_start]
            if len(t) < min_trades:
                continue
            m = t['net'].mean()
            if best is None or m > best['is_avg_net']:
                best = dict(filter=f, stop=stop, tgt=tgt, hold=hold, is_n=len(t), is_avg_net=m)
    return best


def evaluate(coins, pattern, direction, test_start, grid, filters=tuple(FILTERS), cost=DEFAULT_COST, n_perm=300):
    # Tune on the earlier period, then test on the recent period. Returns (row, recent trades) or None.
    test_start = pd.Timestamp(test_start)
    best = tune(coins, pattern, direction, test_start, grid, filters, cost)
    if best is None:
        return None
    mask = eligible_masks(coins, best['filter'], direction)
    t = run_trades(coins, pattern, direction, best['stop'], best['tgt'], best['hold'], cost, eligible=mask)
    rec = t[t['entry'] >= test_start]
    if len(rec) == 0:
        return None
    base = random_entry_baseline(coins, direction, best['stop'], best['tgt'], best['hold'], test_start,
                                 rec.groupby('coin').size().to_dict(), cost, n_perm, eligible=mask)
    st, st25 = trade_stats(rec), trade_stats(rec[rec['entry'] >= '2025-01-01'])
    per_coin = rec.groupby('coin')['net'].mean()
    row = dict(pattern=pattern, side='long' if direction > 0 else 'short', filter=best['filter'],
               rules=f"stop {best['stop'] or '-'} / tgt {best['tgt'] or '-'} / {best['hold']} bars",
               is_n=best['is_n'], is_avg_net_pct=100 * best['is_avg_net'],
               n=st['n'], win_rate=st['win_rate'], avg_net_pct=st['avg_net_pct'],
               profit_factor=st['profit_factor'], worst_pct=st['worst_pct'],
               random_avg_pct=100 * np.nanmean(base),
               p_value=(np.sum(base >= rec['net'].mean()) + 1) / (len(base) + 1),
               n_2025=st25['n'], avg_net_2025_pct=st25['avg_net_pct'],
               coins_pos=int((per_coin > 0).sum()), coins_traded=len(per_coin))
    row['_rules'] = (best['stop'], best['tgt'], best['hold'])
    return row, rec


def judge(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t['q_value'] = benjamini_hochberg(t['p_value'].to_numpy())
    t['proven'] = ((t['n'] >= 30) & (t['q_value'] < 0.10) & (t['avg_net_2025_pct'] > 0) & (t['coins_pos'] >= 4))
    return t


def cross_validate(coins, pattern, direction, train_coins, test_coins, test_start, grid, filters=tuple(FILTERS),
                   cost=DEFAULT_COST, n_perm=300, min_trades=15):
    # Choose filter + exit rules on `train_coins` before test_start, then trade them on DIFFERENT coins
    # from test_start onward. Guards against rules that are tuned to one coin's history.
    test_start = pd.Timestamp(test_start)
    best = tune({s: coins[s] for s in train_coins}, pattern, direction, test_start, grid, filters, cost, min_trades)
    if best is None:
        return None
    te = {s: coins[s] for s in test_coins}
    mask = eligible_masks(te, best['filter'], direction)
    t = run_trades(te, pattern, direction, best['stop'], best['tgt'], best['hold'], cost, eligible=mask)
    rec = t[t['entry'] >= test_start]
    if len(rec) == 0:
        return None
    base = random_entry_baseline(te, direction, best['stop'], best['tgt'], best['hold'], test_start,
                                 rec.groupby('coin').size().to_dict(), cost, n_perm, eligible=mask)
    st = trade_stats(rec)
    return dict(train=','.join(c[:-4] for c in train_coins), test=','.join(c[:-4] for c in test_coins),
                filter=best['filter'], n=st['n'], win_rate=st['win_rate'], avg_net_pct=st['avg_net_pct'],
                random_avg_pct=100 * np.nanmean(base),
                p_value=(np.sum(base >= rec['net'].mean()) + 1) / (len(base) + 1))
