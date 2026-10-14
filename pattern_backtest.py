import argparse
import numpy as np
import pandas as pd

from candlestick_patterns import find_candlestick_patterns
from double_top_bottom import find_multi_tops
from triangles_wedges import find_triangles_wedges
from head_shoulders import find_hs_patterns
from flags_pennants import find_flags_pennants_pips

# Evaluates whether chart patterns predict future returns.
#
# Every detector is turned into a list of events: (signal bar, pattern name, direction).
# The signal bar is the first bar where the pattern is fully known, so entering at
# that bar's close uses no future data. For each event we measure the forward
# log return over several horizons, multiplied by the pattern's direction.
#
# Results are split into an in-sample period and a recent (out-of-sample) period.
# A pattern only looks predictive if it beats random entries in BOTH periods.
# The p-value is from a permutation test: how often do the same number of random
# entries, in the same period and direction, do at least as well?

DEFAULTS = {
    # Settings per bar size: (extreme order, triangle lookback, horizons in bars)
    'daily': dict(order=3, tri_lookback=30, horizons=[1, 5, 10, 20]),
    'hourly': dict(order=6, tri_lookback=48, horizons=[6, 24, 72, 168]),
}


def load_ohlc(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date').sort_index()
    return data[['open', 'high', 'low', 'close']].astype(float)


def collect_events(ohlc: pd.DataFrame, order: int, tri_lookback: int) -> pd.DataFrame:
    log_ohlc = np.log(ohlc)
    close = log_ohlc['close'].to_numpy()
    events = []

    def add(i, name, direction):
        events.append((int(i), name, int(direction)))

    # Candlesticks
    sig = find_candlestick_patterns(ohlc)
    for name in sig.columns:
        for i in np.flatnonzero(sig[name].to_numpy() != 0):
            add(i, name, sig[name].iloc[i])

    # Double / triple tops and bottoms
    for n, label in ((2, 'double'), (3, 'triple')):
        tops, bottoms = find_multi_tops(close, order, n_peaks=n)
        for p in tops:
            add(p.break_i, f'{label}_top', -1)
        for p in bottoms:
            add(p.break_i, f'{label}_bottom', 1)

    # Triangles and wedges, labelled by breakout direction
    for p in find_triangles_wedges(log_ohlc, tri_lookback):
        add(p.break_i, f"{p.shape}_{'up' if p.direction > 0 else 'down'}", p.direction)

    # Existing detectors from this repo
    hs, ihs = find_hs_patterns(close, order)
    for p in hs:
        add(p.break_i, 'head_shoulders', -1)
    for p in ihs:
        add(p.break_i, 'inv_head_shoulders', 1)

    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(close, max(order, 3))
    for pats, name, d in ((bull_flags, 'bull_flag', 1), (bear_flags, 'bear_flag', -1),
                          (bull_pennants, 'bull_pennant', 1), (bear_pennants, 'bear_pennant', -1)):
        for p in pats:
            add(p.conf_x, name, d)

    ev = pd.DataFrame(events, columns=['i', 'pattern', 'direction'])
    ev['date'] = ohlc.index[ev['i']]
    return ev.sort_values('i').reset_index(drop=True)


def forward_returns(close: np.array, horizons) -> dict:
    log_c = np.log(close)
    fwd = {}
    for h in horizons:
        r = np.full(len(log_c), np.nan)
        r[:-h] = log_c[h:] - log_c[:-h]
        fwd[h] = r
    return fwd


def evaluate(events: pd.DataFrame, fwd: dict, period_mask: np.array, period_name: str,
             n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (name, direction), grp in events.groupby(['pattern', 'direction']):
        idx = grp['i'].to_numpy()
        idx = idx[period_mask[idx]]
        for h, r in fwd.items():
            pool = np.flatnonzero(period_mask & ~np.isnan(r))
            sel = idx[~np.isnan(r[idx])]
            n = len(sel)
            if n == 0 or len(pool) == 0:
                continue
            dr = direction * r[sel]
            baseline = direction * r[pool].mean() # Return from entering on random bars
            rand = direction * r[pool][rng.integers(0, len(pool), size=(n_perm, n))].mean(axis=1)
            p_value = (np.sum(rand >= dr.mean()) + 1) / (n_perm + 1)
            rows.append({
                'period': period_name, 'pattern': name, 'direction': direction, 'horizon': h,
                'n': n,
                'hit_rate': (dr > 0).mean(),
                'mean_ret_pct': 100 * dr.mean(),
                'baseline_pct': 100 * baseline,
                'excess_pct': 100 * (dr.mean() - baseline),
                'p_value': p_value,
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, horizon: int, min_n: int, alpha: float) -> pd.DataFrame:
    # One row per pattern at a single horizon: in-sample vs recent side by side.
    r = results[results['horizon'] == horizon]
    cols = ['n', 'hit_rate', 'excess_pct', 'p_value']
    ins = r[r['period'] == 'in_sample'].set_index('pattern')[cols].add_prefix('is_')
    oos = r[r['period'] == 'recent'].set_index('pattern')[cols].add_prefix('recent_')
    s = ins.join(oos, how='outer')
    s[['is_n', 'recent_n']] = s[['is_n', 'recent_n']].fillna(0).astype(int)

    def verdict(row):
        if row['is_n'] < min_n or row['recent_n'] < min_n:
            return 'too few'
        if row['is_excess_pct'] > 0 and row['recent_excess_pct'] > 0 and row['recent_p_value'] < alpha:
            return 'edge holds'
        if row['is_excess_pct'] > 0 and row['recent_excess_pct'] > 0:
            return 'weak'
        if row['is_excess_pct'] > 0 >= row['recent_excess_pct']:
            return 'faded'
        return 'none'

    s['verdict'] = s.apply(verdict, axis=1)
    order = {'edge holds': 0, 'weak': 1, 'faded': 2, 'none': 3, 'too few': 4}
    s['_o'] = s['verdict'].map(order)
    return s.sort_values(['_o', 'recent_excess_pct'], ascending=[True, False]).drop(columns='_o')


def detect_timeframe(index: pd.DatetimeIndex) -> str:
    step = pd.Series(index).diff().median()
    return 'daily' if step >= pd.Timedelta(hours=12) else 'hourly'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest chart pattern predictive power.')
    parser.add_argument('--data', default='BTCUSDT86400.csv')
    parser.add_argument('--test-start', default=None,
                        help='First date of the recent/out-of-sample period (default: last 25%% of data)')
    parser.add_argument('--since', default=None, help='Ignore data before this date')
    parser.add_argument('--horizon', type=int, default=None, help='Horizon (bars) used in the summary table')
    parser.add_argument('--min-n', type=int, default=8, help='Minimum events per period for a verdict')
    parser.add_argument('--alpha', type=float, default=0.1, help='p-value cutoff for "edge holds"')
    parser.add_argument('--out', default='pattern_results.csv')
    args = parser.parse_args()

    ohlc = load_ohlc(args.data)
    if args.since:
        ohlc = ohlc[ohlc.index >= args.since]
    tf = detect_timeframe(ohlc.index)
    cfg = DEFAULTS[tf]

    test_start = pd.Timestamp(args.test_start) if args.test_start else ohlc.index[int(len(ohlc) * 0.75)]
    recent = np.asarray(ohlc.index >= test_start)
    horizon = args.horizon or cfg['horizons'][1]

    print(f'{args.data}: {len(ohlc)} {tf} bars, {ohlc.index[0].date()} -> {ohlc.index[-1].date()}')
    print(f'In-sample: before {test_start.date()} | Recent: {test_start.date()} onward '
          f'({recent.sum()} bars)\n')

    events = collect_events(ohlc, cfg['order'], cfg['tri_lookback'])
    fwd = forward_returns(ohlc['close'].to_numpy(), cfg['horizons'])
    results = pd.concat([
        evaluate(events, fwd, ~recent, 'in_sample'),
        evaluate(events, fwd, recent, 'recent'),
    ], ignore_index=True)
    results.to_csv(args.out, index=False)

    summary = summarize(results, horizon, args.min_n, args.alpha)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print(f'Horizon: {horizon} bars. excess_pct = direction-adjusted return minus random entries.')
    print(summary.round({'is_hit_rate': 2, 'is_excess_pct': 2, 'is_p_value': 3,
                         'recent_hit_rate': 2, 'recent_excess_pct': 2, 'recent_p_value': 3}).to_string())
    print(f'\nAll horizons written to {args.out}')
