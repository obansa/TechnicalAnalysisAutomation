import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from rolling_window import rw_top, rw_bottom


def rsi(close: np.array, period: int = 14) -> np.array:
    # Wilder's RSI
    delta = np.diff(close, prepend=np.nan)
    gain = pd.Series(np.where(delta > 0, delta, 0.0))
    loss = pd.Series(np.where(delta < 0, -delta, 0.0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = (100.0 - 100.0 / (1.0 + rs)).to_numpy().copy()
    out[:period] = np.nan
    return out


def macd_hist(close: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> np.array:
    c = pd.Series(close)
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    hist = macd - macd.ewm(span=signal, adjust=False).mean()
    out = hist.to_numpy().copy()
    out[:slow + signal] = np.nan
    return out


@dataclass
class Divergence:
    kind: str           # 'bull', 'bear', 'hidden_bull' or 'hidden_bear'
    direction: int      # 1 bullish, -1 bearish
    ext1_i: int         # First price extreme (index)
    ext2_i: int         # Second price extreme (index)
    conf_i: int         # Bar where the second extreme is confirmed (signal bar)
    price1: float
    price2: float
    ind1: float
    ind2: float


def find_divergences(price: np.array, indicator: np.array, order: int,
                     max_gap: int = None, hidden: bool = True):
    # Compares each newly confirmed swing low (high) with the previous swing low (high).
    #   Regular bullish: price makes a lower low, indicator a higher low.
    #   Regular bearish: price makes a higher high, indicator a lower high.
    #   Hidden bullish:  price makes a higher low, indicator a lower low (trend continuation).
    #   Hidden bearish:  price makes a lower high, indicator a higher high.
    # The signal bar is where the second swing is confirmed (order bars after it), so no lookahead.
    # Swings further apart than max_gap bars (default 20 * order) are not compared.
    if max_gap is None:
        max_gap = 20 * order

    divs = []
    last_top = None
    last_bot = None
    for i in range(len(price)):
        if rw_bottom(price, i, order):
            k = i - order
            if last_bot is not None and k - last_bot <= max_gap \
                    and not np.isnan(indicator[k]) and not np.isnan(indicator[last_bot]):
                p1, p2 = price[last_bot], price[k]
                v1, v2 = indicator[last_bot], indicator[k]
                if p2 < p1 and v2 > v1:
                    divs.append(Divergence('bull', 1, last_bot, k, i, p1, p2, v1, v2))
                elif hidden and p2 > p1 and v2 < v1:
                    divs.append(Divergence('hidden_bull', 1, last_bot, k, i, p1, p2, v1, v2))
            last_bot = k

        if rw_top(price, i, order):
            k = i - order
            if last_top is not None and k - last_top <= max_gap \
                    and not np.isnan(indicator[k]) and not np.isnan(indicator[last_top]):
                p1, p2 = price[last_top], price[k]
                v1, v2 = indicator[last_top], indicator[k]
                if p2 > p1 and v2 < v1:
                    divs.append(Divergence('bear', -1, last_top, k, i, p1, p2, v1, v2))
                elif hidden and p2 < p1 and v2 > v1:
                    divs.append(Divergence('hidden_bear', -1, last_top, k, i, p1, p2, v1, v2))
            last_top = k

    return divs


def plot_divergence(ohlc: pd.DataFrame, indicator: np.array, div: Divergence,
                    ind_name: str = 'RSI', pad: int = 10):
    start = max(div.ext1_i - pad, 0)
    end = min(div.conf_i + pad, len(ohlc) - 1)
    idx = ohlc.index[start:end + 1]
    color = 'tab:green' if div.direction > 0 else 'tab:red'

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6),
                                   gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(idx, ohlc['close'].iloc[start:end + 1], color='gray', lw=1)
    ax1.plot([ohlc.index[div.ext1_i], ohlc.index[div.ext2_i]],
             [ohlc['close'].iloc[div.ext1_i], ohlc['close'].iloc[div.ext2_i]], color=color, lw=2)
    ax1.axvline(ohlc.index[div.conf_i], color=color, ls=':', lw=1)
    ax1.set_title(f"{div.kind.replace('_', ' ')} divergence ({ind_name})")
    ax2.plot(idx, indicator[start:end + 1], color='gray', lw=1)
    ax2.plot([ohlc.index[div.ext1_i], ohlc.index[div.ext2_i]], [div.ind1, div.ind2], color=color, lw=2)
    ax2.set_ylabel(ind_name)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    close = data['close'].to_numpy()

    divs = find_divergences(np.log(close), rsi(close), order=3)
    print(pd.Series([d.kind for d in divs]).value_counts())
