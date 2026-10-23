import numpy as np
import pandas as pd

from divergences import rsi
from new_patterns import load_tf
from cross_coin import COINS, benjamini_hochberg
from trade_sim import atr, run_trades, trade_stats, random_entry_baseline, DEFAULT_COST

# The rule-based strategies from the VT Markets article "10 Best Trading Strategies for 2026" (3 Aug 2026),
# written as code with the parameters the article states. The article is vague on entries, so every
# definition below is stated explicitly and FIXED BEFORE LOOKING AT RESULTS. Nothing is tuned.
#
#   Trend following  "50 EMA and 200 EMA confirm the direction; only enter on pullbacks within the trend";
#                    R:R 1:2 to 1:3   -> long when EMA50 > EMA200 and the close crosses back above EMA20 after a
#                    close below it (mirror for shorts); stop 2 ATR, target 4 ATR (1:2).
#   Breakout         "break out of a defined range, with rising volume"; example R:R 1:2
#                    -> close beyond the previous N-bar high/low (N = 3 days of bars) with volume > 1.5x its
#                    20-bar average; stop 2 ATR, target 4 ATR (1:2).
#   Range / mean reversion  "Bollinger Bands and RSI with levels at 30 and 70"; confirm range integrity;
#                    R:R 1:1.5 to 1:2  -> long when close < lower band (20, 2) and RSI(14) < 30 (mirror for
#                    shorts); stop 2 ATR, target 3 ATR (1:1.5). A second version requires ADX(14) < 20.
# Entries are at the next bar's open. Cost: 0.15% per side. Strategies the article lists that cannot be tested
# with this data (scalping on 1-5 minute bars, macro position trading, news trading, carry, copy trading) are
# left out; swing trading gives no entry rule.

BARS_PER_3_DAYS = {'daily': 3, '4h': 18, 'hourly': 72}
BREAKOUT_N = {'daily': 20, '4h': 18, 'hourly': 72}   # daily uses a 20-day range; others use 3 days as in the article's example
MR_HOLD = {'daily': 10, '4h': 12, 'hourly': 24}
TREND_HOLD = {'daily': 30, '4h': 60, 'hourly': 120}
BREAKOUT_HOLD = {'daily': 20, '4h': 18, 'hourly': 72}


def load_with_volume(symbol: str, tf: str, data_dir: str = 'data') -> pd.DataFrame:
    secs = 86400 if tf == 'daily' else 3600
    d = pd.read_csv(f'{data_dir}/{symbol}{secs}.csv', parse_dates=['date']).set_index('date').sort_index()
    if tf == '4h':
        d = d.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    return d[['open', 'high', 'low', 'close', 'volume']].astype(float)


def adx(ohlc: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = ohlc['high'], ohlc['low'], ohlc['close']
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus, index=h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def build(ohlc: pd.DataFrame, tf: str):
    # Returns (events DataFrame with columns i, pattern, direction; regime masks {pattern: bool array}).
    c = ohlc['close']
    ema20, ema50, ema200 = (c.ewm(span=s, adjust=False).mean().to_numpy() for s in (20, 50, 200))
    cl = c.to_numpy()
    n = len(cl)
    ev, masks = [], {}

    def add(name, d, sig, regime=None):
        idx = np.flatnonzero(sig)
        ev.extend((int(i), name, d) for i in idx)
        masks[name] = np.ones(n, dtype=bool) if regime is None else regime

    # Trend following: with the trend, entry when price resumes above/below the 20 EMA after a pullback
    up, dn = (ema50 > ema200), (ema50 < ema200)
    prev_below = np.r_[False, cl[:-1] < ema20[:-1]]
    prev_above = np.r_[False, cl[:-1] > ema20[:-1]]
    warm = np.arange(n) >= 200
    add('trend_long', 1, up & prev_below & (cl > ema20) & warm, up & warm)
    add('trend_short', -1, dn & prev_above & (cl < ema20) & warm, dn & warm)

    # Breakout of the previous N-bar range, with and without the volume condition
    N = BREAKOUT_N[tf]
    hi = ohlc['high'].shift(1).rolling(N).max().to_numpy()
    lo = ohlc['low'].shift(1).rolling(N).min().to_numpy()
    volx = (ohlc['volume'] > 1.5 * ohlc['volume'].shift(1).rolling(20).mean()).to_numpy()
    bu, bd = cl > hi, cl < lo
    add('breakout_long_vol', 1, bu & volx)
    add('breakout_short_vol', -1, bd & volx)
    add('breakout_long_novol', 1, bu)
    add('breakout_short_novol', -1, bd)

    # Range / mean reversion: Bollinger(20, 2) + RSI(14) 30/70, with and without a range filter (ADX < 20)
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    upper, lower = (ma + 2 * sd).to_numpy(), (ma - 2 * sd).to_numpy()
    r = rsi(cl)
    a_ = adx(ohlc)
    rng = np.nan_to_num(a_, nan=99) < 20
    ml, ms = (cl < lower) & (r < 30), (cl > upper) & (r > 70)
    add('meanrev_long', 1, ml)
    add('meanrev_short', -1, ms)
    add('meanrev_long_range', 1, ml & rng, rng)
    add('meanrev_short_range', -1, ms & rng, rng)

    e = pd.DataFrame(ev, columns=['i', 'pattern', 'direction']).sort_values('i').reset_index(drop=True)
    e['date'] = ohlc.index[e['i']]
    return e, masks


# stop, target (ATR multiples) and max hold (bars) per strategy family
def rules(pattern: str, tf: str):
    if pattern.startswith('trend'):
        return 2.0, 4.0, TREND_HOLD[tf]
    if pattern.startswith('breakout'):
        return 2.0, 4.0, BREAKOUT_HOLD[tf]
    return 2.0, 3.0, MR_HOLD[tf]


def prepare_vt(symbols, tf: str, data_dir: str = 'data') -> dict:
    out = {}
    for s in symbols:
        o = load_with_volume(s, tf, data_dir)
        e, m = build(o, tf)
        out[s] = dict(ohlc=o, events=e, masks=m, atr=atr(o))
    return out


def evaluate_fixed(coins: dict, pattern: str, direction: int, tf: str, test_start, cost=DEFAULT_COST, n_perm=300):
    # Runs the article's rules unchanged and reports an earlier period and a recent period, each compared with
    # random entries that obey the SAME regime, direction and exit rules.
    test_start = pd.Timestamp(test_start)
    stop, tgt, hold = rules(pattern, tf)
    elig = {s: c['masks'][pattern] for s, c in coins.items()}
    t = run_trades(coins, pattern, direction, stop, tgt, hold, cost, eligible=elig)
    rows = []
    for period, part in (('earlier', t[t['entry'] < test_start]), ('recent', t[t['entry'] >= test_start])):
        if len(part) == 0:
            continue
        per_coin = part.groupby('coin')['net'].agg(['mean', 'size'])
        if period == 'earlier':
            base = random_entry_baseline(coins, direction, stop, tgt, hold, pd.Timestamp('2000-01-01'),
                                         part.groupby('coin').size().to_dict(), cost, n_perm, eligible=elig, pool_end=test_start)
        else:
            base = random_entry_baseline(coins, direction, stop, tgt, hold, test_start,
                                         part.groupby('coin').size().to_dict(), cost, n_perm, eligible=elig)
        st = trade_stats(part)
        rows.append(dict(tf=tf, pattern=pattern, side='long' if direction > 0 else 'short', period=period, n=st['n'],
                         win_rate=st['win_rate'], avg_gross_pct=st['avg_gross_pct'], avg_net_pct=st['avg_net_pct'],
                         profit_factor=st['profit_factor'], random_net_pct=100 * np.nanmean(base),
                         p_value=(np.sum(base >= part['net'].mean()) + 1) / (len(base) + 1),
                         coins_pos=int((per_coin['mean'] > 0).sum()), coins=len(per_coin)))
    return rows, t
