import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from rolling_window import rw_top, rw_bottom
from trade_sim import atr

# Smart-money-concept (SMC / ICT) pattern detectors.
#
# Every detector reports a pattern on the bar where it is first KNOWN, using only that bar and earlier
# bars. tests/test_detectors.py checks this property directly by re-running each detector on truncated
# data and requiring identical events.
#
# Definitions (all fixed in advance; nothing here is tuned on returns):
#   Swing high / low   Rolling-window extreme of `order` bars on each side, confirmed `order` bars later.
#   BOS                Break of structure: a close beyond the latest unbroken swing level, in the direction of
#                      the current trend (up-break in an uptrend, down-break in a downtrend).
#   CHoCH              Change of character: a close beyond the latest swing level AGAINST the current trend.
#   Liquidity sweep    A wick beyond a recent unswept swing level (stop-hunt) that closes back inside.
#   Fair value gap     Three-candle imbalance: bullish if low[i] > high[i-2], bearish if high[i] < low[i-2],
#                      and the gap is at least `min_gap_atr` ATRs.
#   Order block        The last opposing candle (within `ob_lookback` bars) before a displacement candle
#                      (body >= `disp_atr` ATRs) that also breaks structure.
#   Zone retest        First bar after formation (within `max_age` bars) that trades back into the zone and
#                      closes without breaking through it. FVGs and order blocks are signalled both at
#                      formation and at retest.

# Per-timeframe swing order (bars on each side). Chosen from common SMC usage, not from backtest results.
ORDER = {'daily': 4, '4h': 5, 'hourly': 6}


@dataclass
class Zone:
    kind: str           # 'fvg' or 'ob'
    direction: int      # 1 bullish, -1 bearish
    form_i: int         # bar where the zone is known
    top: float
    bottom: float
    retest_i: int = -1  # bar where price first returned into the zone and held (-1 if never)
    invalid_i: int = -1  # bar where price closed through the zone first (-1 if never)


def swings(high: np.ndarray, low: np.ndarray, order: int):
    # {confirmation_bar: (swing_bar, price)} for swing highs and swing lows
    sh, sl = {}, {}
    for i in range(len(high)):
        if rw_top(high, i, order):
            sh[i] = (i - order, high[i - order])
        if rw_bottom(low, i, order):
            sl[i] = (i - order, low[i - order])
    return sh, sl


def find_structure(ohlc: pd.DataFrame, order: int):
    # Returns a list of (bar, kind, direction, broken_level) with kind 'bos' or 'choch'.
    high, low, close = (ohlc[k].to_numpy() for k in ('high', 'low', 'close'))
    sh, sl = swings(high, low, order)
    sh_level = sl_level = None
    trend = 0
    out = []
    for i in range(len(close)):
        if i in sh:
            sh_level = sh[i][1]
        if i in sl:
            sl_level = sl[i][1]
        if sh_level is not None and close[i] > sh_level:
            out.append((i, 'choch' if trend == -1 else 'bos', 1, sh_level))
            trend, sh_level = 1, None
        elif sl_level is not None and close[i] < sl_level:
            out.append((i, 'choch' if trend == 1 else 'bos', -1, sl_level))
            trend, sl_level = -1, None
    return out


def find_liquidity_sweeps(ohlc: pd.DataFrame, order: int, max_levels: int = 3, max_age: int = 200):
    # Returns a list of (bar, direction, swept_level). Bullish sweep: wick below recent swing low(s), close back above.
    high, low, close = (ohlc[k].to_numpy() for k in ('high', 'low', 'close'))
    sh, sl = swings(high, low, order)
    lows, highs = [], []   # active (level, swing_bar), newest last
    out = []
    for i in range(len(close)):
        if i in sl:
            lows.append((sl[i][1], sl[i][0]))
            lows = lows[-max_levels:]
        if i in sh:
            highs.append((sh[i][1], sh[i][0]))
            highs = highs[-max_levels:]
        lows = [(lv, k) for lv, k in lows if i - k <= max_age]
        highs = [(lv, k) for lv, k in highs if i - k <= max_age]

        pierced = [lv for lv, _ in lows if low[i] < lv]
        if pierced:
            if close[i] > max(pierced):
                out.append((i, 1, max(pierced)))
            lows = [(lv, k) for lv, k in lows if lv not in pierced]
        pierced = [lv for lv, _ in highs if high[i] > lv]
        if pierced:
            if close[i] < min(pierced):
                out.append((i, -1, min(pierced)))
            highs = [(lv, k) for lv, k in highs if lv not in pierced]
    return out


def _track_zone(z: Zone, high, low, close, max_age: int):
    # Fill in retest_i / invalid_i for a zone, scanning forward bar by bar.
    for j in range(z.form_i + 1, min(z.form_i + max_age, len(close) - 1) + 1):
        if z.direction > 0:
            if close[j] < z.bottom:
                z.invalid_i = j
                return z
            if low[j] <= z.top:
                z.retest_i = j
                return z
        else:
            if close[j] > z.top:
                z.invalid_i = j
                return z
            if high[j] >= z.bottom:
                z.retest_i = j
                return z
    return z


def find_fvgs(ohlc: pd.DataFrame, min_gap_atr: float = 0.25, max_age: int = 30) -> List[Zone]:
    high, low, close = (ohlc[k].to_numpy() for k in ('high', 'low', 'close'))
    a = atr(ohlc)
    zones = []
    for i in range(2, len(close)):
        if np.isnan(a[i - 1]):
            continue
        if low[i] > high[i - 2] and low[i] - high[i - 2] >= min_gap_atr * a[i - 1]:
            zones.append(_track_zone(Zone('fvg', 1, i, low[i], high[i - 2]), high, low, close, max_age))
        elif high[i] < low[i - 2] and low[i - 2] - high[i] >= min_gap_atr * a[i - 1]:
            zones.append(_track_zone(Zone('fvg', -1, i, low[i - 2], high[i]), high, low, close, max_age))
    return zones


def find_order_blocks(ohlc: pd.DataFrame, order: int, disp_atr: float = 1.5, ob_lookback: int = 6,
                      max_age: int = 30) -> List[Zone]:
    o, high, low, close = (ohlc[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    a = atr(ohlc)
    zones = []
    for i, kind, direction, _ in find_structure(ohlc, order):
        if i < 1 or np.isnan(a[i - 1]):
            continue
        body = (close[i] - o[i]) * direction
        if body < disp_atr * a[i - 1]:
            continue
        for k in range(i - 1, max(i - ob_lookback, 0) - 1, -1):   # last opposing candle before the impulse
            if (close[k] - o[k]) * direction < 0:
                zones.append(_track_zone(Zone('ob', direction, i, high[k], low[k]), high, low, close, max_age))
                break
    return zones


def collect_smc_events(ohlc: pd.DataFrame, order: int, min_gap_atr: float = 0.25, disp_atr: float = 1.5,
                       max_age: int = 30) -> pd.DataFrame:
    # Same schema as pattern_backtest.collect_events: columns i (signal bar), pattern, direction, date.
    ev = []
    side = {1: 'bull', -1: 'bear'}
    for i, kind, d, _ in find_structure(ohlc, order):
        ev.append((i, f'{kind}_{side[d]}', d))
    for i, d, _ in find_liquidity_sweeps(ohlc, order):
        ev.append((i, f'sweep_{side[d]}', d))
    for z in find_fvgs(ohlc, min_gap_atr, max_age):
        ev.append((z.form_i, f'fvg_{side[z.direction]}_form', z.direction))
        if z.retest_i >= 0:
            ev.append((z.retest_i, f'fvg_{side[z.direction]}_retest', z.direction))
    for z in find_order_blocks(ohlc, order, disp_atr, max_age=max_age):
        ev.append((z.form_i, f'ob_{side[z.direction]}_form', z.direction))
        if z.retest_i >= 0:
            ev.append((z.retest_i, f'ob_{side[z.direction]}_retest', z.direction))
    df = pd.DataFrame(ev, columns=['i', 'pattern', 'direction']).sort_values('i').reset_index(drop=True)
    df['date'] = ohlc.index[df['i']]
    return df


def plot_smc(ohlc: pd.DataFrame, start: int, end: int, order: int, ax=None, max_age: int = 30):
    # Candles with fair value gaps (blue/orange boxes, drawn until retested or invalidated), order blocks
    # (green/red boxes), liquidity sweeps (triangles) and BOS/CHoCH levels (dashed lines with labels).
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))
    d = ohlc.iloc[start:end + 1]
    o, h, l, c = (d[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    x = np.arange(start, end + 1)
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, '#1baf7a', '#e34948'), lw=0.8, zorder=2)
    ax.bar(x, np.abs(c - o), bottom=np.minimum(o, c), width=0.6, color=np.where(up, '#1baf7a', '#e34948'), zorder=3)

    def box(z, color, label):
        stop = z.retest_i if z.retest_i >= 0 else (z.invalid_i if z.invalid_i >= 0 else min(z.form_i + max_age, len(ohlc) - 1))
        if stop < start or z.form_i > end:
            return
        x0 = max(z.form_i, start)
        ax.add_patch(Rectangle((x0 - 0.4, z.bottom), min(stop, end) - x0 + 0.8, z.top - z.bottom, color=color, alpha=0.28, zorder=1))
        ax.text(x0, z.top, label, fontsize=7, color=color, va='bottom')

    for z in find_fvgs(ohlc, max_age=max_age):
        box(z, '#2a78d6' if z.direction > 0 else '#eb6834', 'FVG')
    for z in find_order_blocks(ohlc, order, max_age=max_age):
        box(z, '#006b3c' if z.direction > 0 else '#a31515', 'OB')
    for i, d_, lv in find_liquidity_sweeps(ohlc, order):
        if start <= i <= end:
            ax.plot(i, ohlc['low'].iloc[i] if d_ > 0 else ohlc['high'].iloc[i], marker='^' if d_ > 0 else 'v',
                    color='#c98500', ms=9, zorder=5)
    for i, kind, d_, lv in find_structure(ohlc, order):
        if start <= i <= end:
            ax.hlines(lv, max(i - 25, start), i, color='#52514e', ls='--', lw=0.8, zorder=1)
            ax.text(i, lv, kind.upper() + ('+' if d_ > 0 else '-'), fontsize=7, color='#52514e')
    ax.set_xlim(start - 1, end + 1)
    ax.set_ylim(l.min() * 0.995, h.max() * 1.005)
    ax.set_xticks(x[::10]); ax.set_xticklabels([str(ohlc.index[i].date()) + ' ' + str(ohlc.index[i].hour).zfill(2) + 'h' for i in x[::10]], rotation=30, fontsize=7)
    return ax
