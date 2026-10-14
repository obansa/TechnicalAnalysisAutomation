import numpy as np
import pandas as pd

from pattern_backtest import load_ohlc
from cross_coin import COINS
from smart_money import collect_smc_events, ORDER
from cup_handle import cup_handle_events, PARAMS as CUP_PARAMS
from trade_sim import atr

# Detects the new patterns (smart money concepts + cup and handle) on several coins and timeframes and
# packages them in the structure cross_coin.pooled_evaluate / strategy_lab expect:
#   {coin: {'ohlc', 'events', 'fwd', 'recent', 'atr'}}

HORIZONS = {'daily': [1, 5, 10, 20], '4h': [6, 18, 42, 84], 'hourly': [6, 24, 72, 168]}


def load_tf(symbol: str, tf: str, data_dir: str = 'data') -> pd.DataFrame:
    if tf == 'daily':
        return load_ohlc(f'{data_dir}/{symbol}86400.csv')
    h = load_ohlc(f'{data_dir}/{symbol}3600.csv')
    if tf == 'hourly':
        return h
    return h.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()


def detect_all(ohlc: pd.DataFrame, tf: str) -> pd.DataFrame:
    ev = collect_smc_events(ohlc, ORDER[tf])
    cup = cup_handle_events(ohlc, **CUP_PARAMS[tf])
    return pd.concat([ev, cup], ignore_index=True).sort_values('i').reset_index(drop=True)


def prepare_new(symbols, tf: str, test_start: str, data_dir: str = 'data') -> dict:
    test_start = pd.Timestamp(test_start)
    out = {}
    for sym in symbols:
        o = load_tf(sym, tf, data_dir)
        c = np.log(o['close'].to_numpy())
        fwd = {}
        for h in HORIZONS[tf]:
            r = np.full(len(c), np.nan)
            r[:-h] = c[h:] - c[:-h]
            fwd[h] = r
        out[sym] = dict(ohlc=o, events=detect_all(o, tf), fwd=fwd, recent=np.asarray(o.index >= test_start),
                        atr=atr(o))
    return out


def add_momentum_controls(coins: dict, n: int):
    # Control patterns with no SMC logic: first close above the previous n-bar high (long) / below the previous
    # n-bar low (short). Used to check whether a pattern adds anything beyond plain momentum.
    for c in coins.values():
        cl = c['ohlc']['close']
        up = (cl > cl.shift(1).rolling(n).max()).to_numpy()
        dn = (cl < cl.shift(1).rolling(n).min()).to_numpy()
        rows = [(int(i), f'high{n}', 1) for i in np.flatnonzero(up)] + [(int(i), f'low{n}', -1) for i in np.flatnonzero(dn)]
        e = pd.DataFrame(rows, columns=['i', 'pattern', 'direction'])
        e['date'] = c['ohlc'].index[e['i']]
        c['events'] = pd.concat([c['events'], e], ignore_index=True)
