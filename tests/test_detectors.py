import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_money import (find_fvgs, find_order_blocks, find_structure, find_liquidity_sweeps,
                         collect_smc_events)
from cup_handle import find_cup_handles


def candles(rows):
    # rows: (open, high, low, close)
    df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'], dtype=float)
    df.index = pd.date_range('2024-01-01', periods=len(df), freq='D')
    return df


def from_closes(closes, wick=0.1, start=None):
    # Candles whose open is the previous close; small wicks.
    closes = list(map(float, closes))
    o = [closes[0] if start is None else start] + closes[:-1]
    return candles([(op, max(op, c) + wick, min(op, c) - wick, c) for op, c in zip(o, closes)])


def filler(n=30, level=100.0):
    # Quiet flat-ish bars so ATR is defined and small (~1.0)
    rows = []
    for k in range(n):
        c = level + (0.4 if k % 2 else -0.4)
        rows.append((level, level + 0.6, level - 0.6, c))
    return rows


# ------------------------------------------------------------------ fair value gaps
def test_bullish_fvg_zone_and_retest():
    rows = filler() + [(100, 101, 99, 100.5), (100.5, 106, 100, 105), (105, 108, 103, 107),   # gap 101..103
                       (107, 108, 102.5, 106)]                                               # returns into gap
    z = find_fvgs(candles(rows))
    assert len(z) == 1
    f = z[0]
    n = len(filler())
    assert (f.direction, f.form_i, f.bottom, f.top) == (1, n + 2, 101.0, 103.0)
    assert f.retest_i == n + 3 and f.invalid_i == -1


def test_no_fvg_when_candles_overlap():
    rows = filler() + [(100, 101, 99, 100.5), (100.5, 106, 100, 105), (105, 108, 100.8, 107)]   # low <= high[0]
    assert find_fvgs(candles(rows)) == []


def test_fvg_invalidated_by_close_below_zone():
    rows = filler() + [(100, 101, 99, 100.5), (100.5, 106, 100, 105), (105, 108, 103, 107),
                       (107, 107.5, 99.5, 100.2)]    # closes below the gap bottom (101)
    f = find_fvgs(candles(rows))[0]
    assert f.invalid_i == len(filler()) + 3 and f.retest_i == -1


def test_bearish_fvg():
    rows = filler() + [(100, 101, 99, 99.5), (99.5, 100, 94, 95), (95, 97, 92, 93)]      # high[2]=97 < low[0]=99
    f = find_fvgs(candles(rows))[0]
    assert (f.direction, f.bottom, f.top) == (-1, 97.0, 99.0)


def test_tiny_gap_is_ignored():
    rows = filler() + [(100, 101, 99, 100.5), (100.5, 101.5, 100, 101.2), (101.2, 102, 101.05, 101.8)]   # gap 0.05
    assert find_fvgs(candles(rows)) == []


# ------------------------------------------------------------------ structure
def test_bos_then_choch():
    # up-leg, pullback, higher high (BOS), pullback, higher high (BOS), collapse through the last low (CHoCH)
    legs = [(100, 110, 6), (110, 105, 6), (105, 115, 6), (115, 108, 6), (108, 120, 6), (120, 100, 6)]
    path = [100]
    for a, b, n in legs:
        path += list(np.linspace(a, b, n + 1)[1:])
    ev = find_structure(from_closes(path), order=2)
    kinds = [(k, d) for _, k, d, _ in ev]
    assert kinds[:3] == [('bos', 1), ('bos', 1), ('choch', -1)]


def test_structure_break_needs_a_close_not_a_wick():
    path = list(np.linspace(100, 110, 8)) + list(np.linspace(110, 105, 6)[1:]) + [106, 107, 106.5]
    df = from_closes(path)
    df.iloc[-1, df.columns.get_loc('high')] = 112     # wick through 110 but close stays below
    assert [e for e in find_structure(df, order=2) if e[1] == 'bos' and e[2] == 1 and e[0] == len(df) - 1] == []


# ------------------------------------------------------------------ liquidity sweeps
def _swing_low_path():
    # decline to a swing low at 100, rally, then drift
    return list(np.linspace(110, 100, 8)) + list(np.linspace(100, 108, 8)[1:]) + [107.5, 107]


def test_bullish_sweep_wick_below_close_above():
    base = from_closes(_swing_low_path())
    row = (107, 107.2, 98.0, 101.0)                  # wick to 98 (< 100), closes back above 100
    df = candles(base.values.tolist() + [row])
    sw = find_liquidity_sweeps(df, order=2)
    assert (len(df) - 1, 1) in [(i, d) for i, d, _ in sw]


def test_breakdown_is_not_a_sweep():
    base = from_closes(_swing_low_path())
    row = (107, 107.2, 98.0, 99.0)                   # closes below the level: a breakdown
    df = candles(base.values.tolist() + [row])
    assert (len(df) - 1) not in [i for i, _, _ in find_liquidity_sweeps(df, order=2)]


def test_sweep_fires_once_per_level():
    base = from_closes(_swing_low_path()).values.tolist()
    df = candles(base + [(107, 107.2, 98.0, 101.0), (101, 102, 97.0, 101.5)])   # second wick: level already swept
    hits = [i for i, d, _ in find_liquidity_sweeps(df, order=2) if d == 1]
    assert hits == [len(base)]


# ------------------------------------------------------------------ order blocks
def test_bullish_order_block_is_last_down_candle_before_impulse():
    closes = list(np.linspace(100, 110, 11)) + [109, 108, 107, 106]
    base = from_closes(closes).values.tolist()
    last_down = base[-1]                              # candle closing 107 -> 106
    impulse = (106.0, 113.5, 105.9, 113.0)            # displacement through the 110 swing high
    df = candles(base + [impulse])
    z = find_order_blocks(df, order=2)
    assert len(z) == 1
    assert (z[0].direction, z[0].form_i) == (1, len(df) - 1)
    assert (z[0].top, z[0].bottom) == (last_down[1], last_down[2])


def test_small_candle_break_is_not_an_order_block():
    closes = list(np.linspace(100, 110, 11)) + [109, 108, 107, 106, 106.5, 107, 108, 109, 110.3]
    assert find_order_blocks(from_closes(closes), order=2) == []


# ------------------------------------------------------------------ no lookahead (all detectors)
def _random_walk(n, seed, vol=0.02):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c})
    df.index = pd.date_range('2020-01-01', periods=n, freq='D')
    return df


def _real_slice():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ETHUSDT86400.csv')
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, parse_dates=['date']).set_index('date')[['open', 'high', 'low', 'close']]
    return d.iloc[1000:2800]


@pytest.mark.parametrize('name', ['random', 'real'])
def test_smc_events_do_not_use_future_bars(name):
    df = _random_walk(1500, 1) if name == 'random' else _real_slice()
    if df is None:
        pytest.skip('no data')
    full = collect_smc_events(df, order=4)
    key = lambda e: set(zip(e['i'], e['pattern'], e['direction']))
    for cut in (500, 800, 1100, len(df) - 1):
        part = collect_smc_events(df.iloc[:cut + 1], order=4)
        assert key(part) == key(full[full['i'] <= cut]), f'events differ when data is cut at {cut}'


@pytest.mark.parametrize('name', ['random', 'real'])
def test_cup_handle_does_not_use_future_bars(name):
    df = _random_walk(2500, 3, vol=0.03) if name == 'random' else _real_slice()
    if df is None:
        pytest.skip('no data')
    c = np.log(df['close'].to_numpy())
    full = find_cup_handles(c, order=5)
    for cut in (600, 1000, 1500, len(c) - 1):
        part = find_cup_handles(c[:cut + 1], order=5)
        assert [(p.left_i, p.right_i, p.break_i) for p in part] == \
               [(p.left_i, p.right_i, p.break_i) for p in full if p.break_i <= cut]


# ------------------------------------------------------------------ cup and handle
def _cup_path(width=60, depth=0.25, handle_bars=8, handle_depth=0.05, rise_bars=6, noise=0.002, seed=0,
              lead=30, v_shape=False, right_rim=0.0):
    rng = np.random.default_rng(seed)
    lead_in = np.linspace(-0.3, 0.0, lead)
    x = np.linspace(0, 1, width + 1)
    cup = -depth * (1 - np.abs(2 * x - 1)) if v_shape else -depth * (1 - (2 * x - 1) ** 2)
    cup = cup + x * right_rim                        # tilt so the right rim sits at `right_rim`
    handle = np.linspace(right_rim, right_rim - handle_depth, handle_bars + 1)[1:]
    back = np.linspace(right_rim - handle_depth, right_rim + 0.04, rise_bars + 1)[1:]
    after = right_rim + 0.04 + 0.01 * np.arange(1, 11)
    path = np.r_[lead_in, cup[1:] if False else cup, handle, back, after]
    path = path + rng.normal(0, noise, len(path))
    return path, lead, lead + width


def test_detects_planted_cup_and_handle():
    path, left, right = _cup_path()
    found = find_cup_handles(path, order=5)
    assert len(found) == 1
    p = found[0]
    assert abs(p.left_i - left) <= 3 and abs(p.right_i - right) <= 3
    assert right < p.break_i <= right + 8 + 6 + 2
    assert 0.2 < p.depth < 0.3 and p.r2 > 0.9
    assert p.target > p.right_p and p.stop < p.right_p


@pytest.mark.parametrize('kwargs,why', [
    (dict(v_shape=True), 'V-shaped bottom'),
    (dict(handle_depth=0.17), 'handle retraces more than half the cup'),
    (dict(width=18), 'cup too narrow'),
    (dict(depth=0.04), 'cup too shallow'),
    (dict(right_rim=-0.12), 'right rim far below the left rim'),
    (dict(handle_bars=2, handle_depth=0.03), 'handle too short'),
])
def test_rejects_malformed_cups(kwargs, why):
    path, _, _ = _cup_path(**kwargs)
    assert find_cup_handles(path, order=5) == [], why


def test_random_walks_rarely_produce_cups():
    total_bars, total = 0, 0
    for seed in range(40):
        c = np.log(_random_walk(3000, 100 + seed, vol=0.03)['close'].to_numpy())
        total += len(find_cup_handles(c, order=5))
        total_bars += len(c)
    assert total / total_bars * 1000 < 1.0, f'{total} cups in {total_bars} random bars'
