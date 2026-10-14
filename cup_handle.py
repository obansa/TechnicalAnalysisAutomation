import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

from rolling_window import rw_top

# Cup and handle (bullish continuation).
#
#   left rim --> rounded bottom --> right rim --> small pullback (handle) --> close above the right rim
#
# A pattern is reported at the bar whose close first rises above the right rim ("breakout bar"), using only
# data up to that bar. All prices are log closing prices (use np.log(close)).
#
# Rules (fixed in advance from the textbook definition; not tuned on returns):
#   Rims          Both are confirmed swing highs (`order` bars each side); prices within `rim_tol` of each other.
#   Cup width     cup_min..cup_max bars between the rims.
#   Cup depth     depth_min..depth_max below the lower rim (log units: 0.10 ~ a 10% drop, 0.45 ~ a 36% drop).
#   Rounded       A parabola fitted to the cup opens upward with R^2 >= min_r2, the bottom lies in the middle
#                 part of the cup (bottom_pos), and at least min_lower_half_frac of the cup's bars close in the
#                 lower half of its depth (a V-shape spends only ~50% of its time there, a U-shape ~70%+).
#   Clean cup     No close inside the cup rises more than inside_tol above the higher rim.
#   Handle        Lasts handle_min .. handle_max_frac*cup width bars, retraces at least handle_depth_min and at
#                 most handle_depth_max of the cup's depth, so it stays in the upper part of the cup.
#   Measured move target = right rim + cup depth;  stop = lowest close of the handle.


@dataclass
class CupHandle:
    left_i: int
    bottom_i: int
    right_i: int
    handle_low_i: int
    break_i: int
    left_p: float
    right_p: float
    bottom_p: float
    handle_low_p: float
    break_p: float
    depth: float
    handle_depth: float
    r2: float
    target: float
    stop: float

    @property
    def width(self) -> int:
        return self.right_i - self.left_i


def _cup_quality(y: np.ndarray, rim_low: float, depth_min: float, depth_max: float, min_r2: float,
                 bottom_pos, min_lower_half_frac: float, inside_tol: float, rim_high: float):
    # Returns (ok, r2, bottom_offset, depth). y = closes from the left rim to the right rim inclusive.
    n = len(y)
    bottom_off = int(np.argmin(y))
    depth = rim_low - y[bottom_off]
    if not (depth_min <= depth <= depth_max):
        return False, 0.0, bottom_off, depth
    if y.max() > rim_high + inside_tol:
        return False, 0.0, bottom_off, depth
    if not (bottom_pos[0] <= bottom_off / (n - 1) <= bottom_pos[1]):
        return False, 0.0, bottom_off, depth
    x = np.linspace(0.0, 1.0, n)
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return False, 0.0, bottom_off, depth
    fit = a * x * x + b * x + c
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ((y - fit) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    if r2 < min_r2:
        return False, r2, bottom_off, depth
    lower_half = (y < y[bottom_off] + depth / 2.0).mean()
    if lower_half < min_lower_half_frac:
        return False, r2, bottom_off, depth
    return True, r2, bottom_off, depth


def find_cup_handles(close: np.ndarray, order: int = 5, cup_min: int = 30, cup_max: int = 150,
                     rim_tol: float = 0.07, depth_min: float = 0.10, depth_max: float = 0.45,
                     min_r2: float = 0.7, bottom_pos=(0.3, 0.7), min_lower_half_frac: float = 0.6,
                     inside_tol: float = 0.03, handle_min: int = 5, handle_max_frac: float = 0.5,
                     handle_depth_min: float = 0.02, handle_depth_max: float = 0.5) -> List[CupHandle]:
    n = len(close)
    swing_highs = []       # confirmed swing highs: bar indices
    pending = []           # right-rim candidates waiting for a breakout
    found: List[CupHandle] = []
    handle_max = int(cup_max * handle_max_frac)

    for i in range(n):
        if rw_top(close, i, order):
            swing_highs.append(i - order)
            pending.append(i - order)

        # Breakout bars: first close above a pending right rim
        best = None
        still = []
        for r in pending:
            p_r = close[r]
            if close[i] > p_r:
                cand = _evaluate(close, swing_highs, r, i, cup_min, cup_max, rim_tol, depth_min, depth_max, min_r2,
                                 bottom_pos, min_lower_half_frac, inside_tol, handle_min, handle_max_frac,
                                 handle_depth_min, handle_depth_max)
                if cand is not None and (best is None or cand.r2 > best.r2):
                    best = cand
            elif i - r <= handle_max:
                still.append(r)
        pending = still
        if best is not None and (not found or best.left_i >= found[-1].right_i):
            found.append(best)
    return found


def _evaluate(close, swing_highs, r, b, cup_min, cup_max, rim_tol, depth_min, depth_max, min_r2, bottom_pos,
              min_lower_half_frac, inside_tol, handle_min, handle_max_frac, handle_depth_min, handle_depth_max):
    p_r = close[r]
    handle = close[r + 1:b]
    if len(handle) < handle_min:
        return None
    best = None
    for l in swing_highs:
        width = r - l
        if width < cup_min:
            continue
        if width > cup_max:
            continue
        p_l = close[l]
        if abs(p_l - p_r) > rim_tol:
            continue
        if len(handle) > handle_max_frac * width:
            continue
        rim_low, rim_high = min(p_l, p_r), max(p_l, p_r)
        ok, r2, boff, depth = _cup_quality(close[l:r + 1], rim_low, depth_min, depth_max, min_r2, bottom_pos,
                                           min_lower_half_frac, inside_tol, rim_high)
        if not ok:
            continue
        h_low = handle.min()
        h_depth = p_r - h_low
        if not (handle_depth_min <= h_depth <= handle_depth_max * depth):
            continue
        if best is None or r2 > best.r2:
            best = CupHandle(left_i=l, bottom_i=l + boff, right_i=r, handle_low_i=r + 1 + int(np.argmin(handle)),
                             break_i=b, left_p=p_l, right_p=p_r, bottom_p=close[l + boff], handle_low_p=h_low,
                             break_p=close[b], depth=depth, handle_depth=h_depth, r2=r2,
                             target=p_r + depth, stop=h_low)
    return best


def cup_handle_events(ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Same schema as pattern_backtest.collect_events (i = breakout bar).
    pats = find_cup_handles(np.log(ohlc['close'].to_numpy()), **kwargs)
    df = pd.DataFrame({'i': [p.break_i for p in pats], 'pattern': 'cup_handle', 'direction': 1})
    df['date'] = ohlc.index[df['i']] if len(df) else pd.DatetimeIndex([])
    return df


# Per-timeframe parameters: bar counts scale with the bar size so a "cup" spans roughly the same calendar time.
PARAMS = {
    'daily': dict(order=5, cup_min=30, cup_max=150, handle_min=5),
    '4h': dict(order=8, cup_min=100, cup_max=500, handle_min=15),
    'hourly': dict(order=12, cup_min=300, cup_max=1500, handle_min=40),
}


def plot_cup_handle(ohlc: pd.DataFrame, pat: CupHandle, ax=None, pad_frac: float = 0.25, after: int = 15):
    pad = int(pat.width * pad_frac)
    s, e = max(pat.left_i - pad, 0), min(pat.break_i + after, len(ohlc) - 1)
    c = ohlc['close'].iloc[s:e + 1]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(c.index, c.values, color='#8a8984', lw=1.2)
    idx = ohlc.index
    rim = np.exp(pat.right_p)
    ax.plot([idx[pat.left_i], idx[pat.right_i]], [np.exp(pat.left_p), rim], color='#2a78d6', lw=1.6)
    ax.plot(idx[[pat.left_i, pat.bottom_i, pat.right_i, pat.handle_low_i, pat.break_i]],
            np.exp([pat.left_p, pat.bottom_p, pat.right_p, pat.handle_low_p, pat.break_p]),
            'o', color='#eb6834', ms=5)
    ax.axhline(rim, color='#2a78d6', ls=':', lw=1)
    ax.axhline(np.exp(pat.target), color='#1baf7a', ls='--', lw=1)
    ax.set_title(f'{idx[pat.left_i].date()} -> {idx[pat.break_i].date()}  drop {100 * (1 - np.exp(-pat.depth)):.0f}%  R2 {pat.r2:.2f}', fontsize=9)
    ax.tick_params(labelsize=7)
    return ax
