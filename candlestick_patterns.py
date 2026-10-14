import pandas as pd
import numpy as np

# Each pattern is evaluated on the bar where it completes (the signal bar), using
# only that bar and earlier bars. Bullish patterns return +1, bearish -1, else 0.
# "Trend" context is the log return over the previous trend_lookback bars, so a
# reversal pattern only counts if it actually follows a move in the other direction.


def _parts(ohlc: pd.DataFrame):
    o = ohlc['open'].to_numpy()
    h = ohlc['high'].to_numpy()
    l = ohlc['low'].to_numpy()
    c = ohlc['close'].to_numpy()
    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return o, h, l, c, body, rng, upper, lower


def _shift(a: np.array, n: int):
    out = np.full_like(a, np.nan, dtype=float)
    if n < len(a):
        out[n:] = a[:-n] if n > 0 else a
    return out


def _shift_bool(a: np.array, n: int):
    # Shifted boolean array, False where there is no earlier bar
    return _shift(a.astype(float), n) == 1.0


def find_candlestick_patterns(ohlc: pd.DataFrame, trend_lookback: int = 5,
                              doji_thresh: float = 0.1, shadow_mult: float = 2.0) -> pd.DataFrame:
    # Returns a DataFrame (same index as ohlc) with one column per pattern
    # holding +1 (bullish), -1 (bearish) or 0.
    o, h, l, c, body, rng, upper, lower = _parts(ohlc)
    rng_safe = np.where(rng > 0, rng, np.nan)
    avg_body = pd.Series(body).rolling(20, min_periods=5).mean().to_numpy()

    log_c = np.log(c)
    trend = log_c - _shift(log_c, trend_lookback)
    prior_trend = _shift(trend, 1) # Trend leading into the signal bar
    down = prior_trend < 0
    up = prior_trend > 0

    bull = c > o
    bear = c < o
    o1, c1, body1 = _shift(o, 1), _shift(c, 1), _shift(body, 1)
    o2, c2, body2 = _shift(o, 2), _shift(c, 2), _shift(body, 2)
    bull1, bear1 = c1 > o1, c1 < o1
    bull2, bear2 = c2 > o2, c2 < o2

    out = {}

    # Engulfing: current body fully covers the previous opposite-colored body
    out['bullish_engulfing'] = (bull & bear1 & (c >= o1) & (o <= c1) & (body > body1) & down)
    out['bearish_engulfing'] = (bear & bull1 & (c <= o1) & (o >= c1) & (body > body1) & up)

    # Hammer / hanging man: long lower shadow, small upper shadow
    hammer_shape = (lower >= shadow_mult * body) & (upper <= body + 0.1 * rng) & (body > 0)
    out['hammer'] = hammer_shape & down
    out['hanging_man'] = hammer_shape & up

    # Shooting star / inverted hammer: long upper shadow, small lower shadow
    star_shape = (upper >= shadow_mult * body) & (lower <= body + 0.1 * rng) & (body > 0)
    out['shooting_star'] = star_shape & up
    out['inverted_hammer'] = star_shape & down

    # Doji after a trend: indecision, read as a reversal signal
    doji = (body / rng_safe) < doji_thresh
    out['doji_bull'] = doji & down
    out['doji_bear'] = doji & up

    # Morning / evening star: big candle, small-bodied middle candle, strong opposite candle
    small1 = body1 < 0.5 * _shift(avg_body, 1)
    big2 = body2 > _shift(avg_body, 2)
    mid2 = (o2 + c2) / 2
    out['morning_star'] = bear2 & big2 & small1 & bull & (c > mid2) & _shift_bool(down, 2)
    out['evening_star'] = bull2 & big2 & small1 & bear & (c < mid2) & _shift_bool(up, 2)

    # Three white soldiers / black crows: three strong candles closing progressively further
    strong = body > 0.5 * rng
    strong1 = _shift_bool(strong, 1)
    strong2 = _shift_bool(strong, 2)
    out['three_white_soldiers'] = bull & bull1 & bull2 & strong & strong1 & strong2 & (c > c1) & (c1 > c2)
    out['three_black_crows'] = bear & bear1 & bear2 & strong & strong1 & strong2 & (c < c1) & (c1 < c2)

    direction = {
        'bullish_engulfing': 1, 'bearish_engulfing': -1,
        'hammer': 1, 'hanging_man': -1,
        'shooting_star': -1, 'inverted_hammer': 1,
        'doji_bull': 1, 'doji_bear': -1,
        'morning_star': 1, 'evening_star': -1,
        'three_white_soldiers': 1, 'three_black_crows': -1,
    }
    sig = pd.DataFrame(index=ohlc.index)
    for name, mask in out.items():
        mask = np.nan_to_num(np.asarray(mask, dtype=float), nan=0.0).astype(bool)
        sig[name] = np.where(mask, direction[name], 0)
    return sig


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    sig = find_candlestick_patterns(data)
    print((sig != 0).sum())
