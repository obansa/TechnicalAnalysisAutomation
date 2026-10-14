import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List
from rolling_window import rw_top, rw_bottom


@dataclass
class MultiTopPattern:
    # True for double/triple bottom (bullish), False for double/triple top (bearish)
    bottom: bool
    n_peaks: int                # 2 = double, 3 = triple

    peaks: List[int] = field(default_factory=list)      # Indices of the tops (or bottoms)
    peaks_p: List[float] = field(default_factory=list)
    troughs: List[int] = field(default_factory=list)    # Indices of the extremes between peaks
    troughs_p: List[float] = field(default_factory=list)

    neckline: float = -1.       # Price that must be broken to confirm
    height: float = -1.         # Distance from peaks to neckline
    width: int = -1             # Bars from first to last peak

    break_i: int = -1           # Index where the neckline is broken (signal bar)
    break_p: float = -1.
    target: float = -1.         # Measured move target
    stop: float = -1.           # Invalidation price


def _check_pattern(ext_i: List[int], ext_t: List[int], data: np.array, n_peaks: int,
                   bottom: bool, peak_tol: float, min_height: float):
    # ext_i / ext_t hold the last 2 * n_peaks - 1 extremes (indices, types).
    # For tops the sequence must be top, bottom, top (, bottom, top).
    need = -1 if bottom else 1
    for k, t in enumerate(ext_t):
        expected = need if k % 2 == 0 else -need
        if t != expected:
            return None

    peaks = ext_i[0::2]
    troughs = ext_i[1::2]
    peaks_p = data[peaks]
    troughs_p = data[troughs]

    if bottom:
        neckline = troughs_p.max()
        height = neckline - peaks_p.min()
    else:
        neckline = troughs_p.min()
        height = peaks_p.max() - neckline

    if height < min_height:
        return None

    # Peaks must be at similar levels relative to the pattern height
    if (peaks_p.max() - peaks_p.min()) > peak_tol * height:
        return None

    pat = MultiTopPattern(bottom, n_peaks)
    pat.peaks = list(peaks)
    pat.peaks_p = list(peaks_p)
    pat.troughs = list(troughs)
    pat.troughs_p = list(troughs_p)
    pat.neckline = neckline
    pat.height = height
    pat.width = peaks[-1] - peaks[0]
    if bottom:
        pat.stop = peaks_p.min()
        pat.target = neckline + height
    else:
        pat.stop = peaks_p.max()
        pat.target = neckline - height
    return pat


def find_multi_tops(data: np.array, order: int, n_peaks: int = 2,
                    peak_tol: float = 0.25, min_height: float = 0.0):
    # Finds double (n_peaks=2) or triple (n_peaks=3) tops and bottoms.
    # data should be log closing prices. A pattern is only reported once
    # price closes through the neckline, and break_i is that bar, so there is no lookahead.
    # A pending pattern is dropped if price moves past the peaks first, or if the
    # neckline is not broken within the pattern's own width (at least 2 * order bars).
    assert order >= 1 and n_peaks in (2, 3)
    n_ext = 2 * n_peaks - 1

    ext_i = []
    ext_t = []
    pending_top = None
    pending_bot = None
    tops = []
    bottoms = []
    last_used_top = -1
    last_used_bot = -1

    for i in range(len(data)):
        new_ext = False
        if rw_top(data, i, order):
            ext_i.append(i - order)
            ext_t.append(1)
            new_ext = True
        if rw_bottom(data, i, order):
            ext_i.append(i - order)
            ext_t.append(-1)
            new_ext = True

        if new_ext and len(ext_i) >= n_ext:
            last_i = ext_i[-n_ext:]
            last_t = ext_t[-n_ext:]
            if last_t[-1] == 1 and last_i[0] > last_used_top:
                pat = _check_pattern(last_i, last_t, data, n_peaks, False, peak_tol, min_height)
                if pat is not None:
                    pending_top = pat
            if last_t[-1] == -1 and last_i[0] > last_used_bot:
                pat = _check_pattern(last_i, last_t, data, n_peaks, True, peak_tol, min_height)
                if pat is not None:
                    pending_bot = pat

        if pending_top is not None:
            if data[i] > pending_top.stop or i - pending_top.peaks[-1] > max(pending_top.width, 2 * order):
                pending_top = None
            elif data[i] < pending_top.neckline:
                pending_top.break_i = i
                pending_top.break_p = data[i]
                tops.append(pending_top)
                last_used_top = pending_top.peaks[-1]
                pending_top = None

        if pending_bot is not None:
            if data[i] < pending_bot.stop or i - pending_bot.peaks[-1] > max(pending_bot.width, 2 * order):
                pending_bot = None
            elif data[i] > pending_bot.neckline:
                pending_bot.break_i = i
                pending_bot.break_p = data[i]
                bottoms.append(pending_bot)
                last_used_bot = pending_bot.peaks[-1]
                pending_bot = None

    return tops, bottoms


def plot_multi_top(candle_data: pd.DataFrame, pat: MultiTopPattern, pad: int = 3):
    import mplfinance as mpf
    start = max(pat.peaks[0] - pat.width // 2 - pad, 0)
    end = min(pat.break_i + pad, len(candle_data) - 1)
    dat = candle_data.iloc[start:end + 1]
    idx = candle_data.index

    pts = sorted(list(zip(pat.peaks, pat.peaks_p)) + list(zip(pat.troughs, pat.troughs_p)))
    path = [(idx[k], np.exp(p)) for k, p in pts] + [(idx[pat.break_i], np.exp(pat.break_p))]
    neck = [(idx[pat.troughs[0]], np.exp(pat.neckline)), (idx[pat.break_i], np.exp(pat.neckline))]

    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()
    name = ('Triple ' if pat.n_peaks == 3 else 'Double ') + ('Bottom' if pat.bottom else 'Top')
    mpf.plot(dat, alines=dict(alines=[path, neck], colors=['w', 'r']), type='candle', style='charles', ax=ax)
    ax.set_title(name)
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    close = np.log(data['close'].to_numpy())

    for n in (2, 3):
        tops, bottoms = find_multi_tops(close, order=6, n_peaks=n)
        print(f'{n} peaks: {len(tops)} tops, {len(bottoms)} bottoms')
