import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from trendline_automation import fit_trendlines_high_low

SHAPES = ['ascending_triangle', 'descending_triangle', 'symmetric_triangle',
          'rising_wedge', 'falling_wedge']


@dataclass
class ConvergingPattern:
    shape: str              # One of SHAPES
    start_i: int            # First bar of the window the lines were fit on
    end_i: int              # Last bar of the window the lines were fit on
    break_i: int            # Bar whose close broke out of the pattern (signal bar)
    break_p: float
    direction: int          # 1 = broke above resistance, -1 = broke below support

    support_slope: float
    support_intercept: float    # Intercept relative to start_i
    resist_slope: float
    resist_intercept: float

    start_width: float      # Distance between lines at start_i
    end_width: float        # Distance between lines at end_i


def _count_touches(line: np.array, y: np.array, tol: float, min_gap: int) -> int:
    # Number of separated places where price comes within tol of the line
    near = np.where(np.abs(line - y) <= tol)[0]
    if len(near) == 0:
        return 0
    return 1 + int((np.diff(near) >= min_gap).sum())


def classify_lines(support_slope: float, resist_slope: float, start_width: float,
                   end_width: float, bar_scale: float,
                   flat_thresh: float = 0.1, min_converge: float = 0.3):
    # Slopes are normalized by bar_scale (average bar range), so flat_thresh means
    # "less than 10% of a typical bar's range per bar".
    if start_width <= 0 or end_width <= 0:
        return None
    if end_width > (1.0 - min_converge) * start_width:
        return None # Not converging enough

    s = support_slope / bar_scale
    r = resist_slope / bar_scale
    s_flat = abs(s) < flat_thresh
    r_flat = abs(r) < flat_thresh

    if r_flat and s > 0:
        return 'ascending_triangle'
    if s_flat and r < 0:
        return 'descending_triangle'
    if r < 0 and s > 0:
        return 'symmetric_triangle'
    if r > 0 and s > r:
        return 'rising_wedge'
    if s < 0 and r < s:
        return 'falling_wedge'
    return None


def find_triangles_wedges(ohlc: pd.DataFrame, lookback: int, step: int = 1,
                          flat_thresh: float = 0.1, min_converge: float = 0.3,
                          min_touches: int = 2, max_hold: int = None):
    # ohlc must hold log prices in columns high, low, close.
    # Every `step` bars, fit support/resistance on the last `lookback` bars.
    # If the lines converge into a recognized shape, the pattern stays active for
    # up to max_hold bars (default lookback // 2); the first close outside the lines
    # in that period is the breakout and is reported at that bar (no lookahead).
    high = ohlc['high'].to_numpy()
    low = ohlc['low'].to_numpy()
    close = ohlc['close'].to_numpy()
    if max_hold is None:
        max_hold = lookback // 2
    min_gap = max(lookback // 10, 2)

    patterns = []
    active = None
    last_end = -1
    for i in range(lookback - 1, len(close)):

        # Check active pattern for breakout
        if active is not None:
            x = i - active.start_i
            resist = active.resist_intercept + active.resist_slope * x
            support = active.support_intercept + active.support_slope * x
            if close[i] > resist or close[i] < support:
                active.break_i = i
                active.break_p = close[i]
                active.direction = 1 if close[i] > resist else -1
                patterns.append(active)
                last_end = i
                active = None
            elif i - active.end_i > max_hold:
                active = None

        # Look for a new pattern (not overlapping a used/active one)
        if active is not None or (i - lookback + 1) <= last_end or i % step != 0:
            continue

        start = i - lookback + 1
        h = high[start:i + 1]
        l = low[start:i + 1]
        c = close[start:i + 1]
        try:
            (s_slope, s_int), (r_slope, r_int) = fit_trendlines_high_low(h, l, c)
        except Exception:
            continue

        start_width = r_int - s_int
        end_width = (r_int + r_slope * (lookback - 1)) - (s_int + s_slope * (lookback - 1))
        bar_scale = np.mean(h - l)
        if bar_scale <= 0:
            continue

        shape = classify_lines(s_slope, r_slope, start_width, end_width, bar_scale,
                               flat_thresh, min_converge)
        if shape is None:
            continue

        x = np.arange(lookback)
        tol = 0.5 * bar_scale
        if _count_touches(s_int + s_slope * x, l, tol, min_gap) < min_touches:
            continue
        if _count_touches(r_int + r_slope * x, h, tol, min_gap) < min_touches:
            continue

        active = ConvergingPattern(shape, start, i, -1, -1., 0,
                                   s_slope, s_int, r_slope, r_int, start_width, end_width)

    return patterns


def plot_converging(candle_data: pd.DataFrame, pat: ConvergingPattern, pad: int = 3):
    import mplfinance as mpf
    start = max(pat.start_i - pad, 0)
    end = min(pat.break_i + pad, len(candle_data) - 1)
    dat = candle_data.iloc[start:end + 1]
    idx = candle_data.index
    x_end = pat.break_i - pat.start_i

    s_line = [(idx[pat.start_i], np.exp(pat.support_intercept)),
              (idx[pat.break_i], np.exp(pat.support_intercept + pat.support_slope * x_end))]
    r_line = [(idx[pat.start_i], np.exp(pat.resist_intercept)),
              (idx[pat.break_i], np.exp(pat.resist_intercept + pat.resist_slope * x_end))]

    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()
    mpf.plot(dat, alines=dict(alines=[s_line, r_line], colors=['g', 'r']), type='candle', style='charles', ax=ax)
    ax.set_title(f"{pat.shape.replace('_', ' ').title()} - breakout {'up' if pat.direction > 0 else 'down'}")
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    pats = find_triangles_wedges(np.log(data[['open', 'high', 'low', 'close']]), lookback=30)
    summary = pd.DataFrame([(p.shape, p.direction) for p in pats], columns=['shape', 'direction'])
    print(summary.groupby(['shape', 'direction']).size())
