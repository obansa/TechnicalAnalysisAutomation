import numpy as np
import pandas as pd

from pattern_backtest import load_ohlc, collect_events, forward_returns, DEFAULTS

# Runs every pattern detector on several coins and pools the results, so each pattern
# is judged on many more signals than a single coin can give. A pattern only looks real
# if it beats random entries across coins, not just on one.

COINS = ['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']


def load_coin(symbol: str, tf: str, data_dir: str = 'data') -> pd.DataFrame:
    secs = 86400 if tf == 'daily' else 3600
    return load_ohlc(f'{data_dir}/{symbol}{secs}.csv')


def prepare(symbols, tf: str, test_start: str, data_dir: str = 'data') -> dict:
    # Detect patterns once per coin; keeps what the evaluation needs.
    cfg = DEFAULTS[tf]
    test_start = pd.Timestamp(test_start)
    out = {}
    for sym in symbols:
        ohlc = load_coin(sym, tf, data_dir)
        events = collect_events(ohlc, cfg['order'], cfg['tri_lookback'])
        out[sym] = dict(
            ohlc=ohlc, events=events,
            fwd=forward_returns(ohlc['close'].to_numpy(), cfg['horizons']),
            recent=np.asarray(ohlc.index >= test_start),
        )
    return out


def pooled_evaluate(coins: dict, horizons, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    # One row per (period, pattern, horizon): signals from all coins combined.
    # p_value: how often the same number of random entries (same coins, same period,
    # same direction) do at least as well. n_coins_pos: coins where the pattern beat random.
    rng = np.random.default_rng(seed)
    all_events = pd.concat([c['events'].assign(coin=s) for s, c in coins.items()], ignore_index=True)
    rows = []
    for period in ('in_sample', 'recent'):
        for (name, direction), grp in all_events.groupby(['pattern', 'direction']):
            for h in horizons:
                obs, base_sum, n_tot = [], 0.0, 0
                rand_sum = np.zeros(n_perm)
                per_coin_pos = 0
                n_coins = 0
                for sym, g in grp.groupby('coin'):
                    c = coins[sym]
                    mask = c['recent'] if period == 'recent' else ~c['recent']
                    r = c['fwd'][h]
                    idx = g['i'].to_numpy()
                    idx = idx[mask[idx]]
                    idx = idx[~np.isnan(r[idx])]
                    if len(idx) == 0:
                        continue
                    pool = np.flatnonzero(mask & ~np.isnan(r))
                    dr = direction * r[idx]
                    base = direction * r[pool].mean()
                    obs.append(dr)
                    base_sum += base * len(idx)
                    n_tot += len(idx)
                    rand_sum += direction * r[pool][rng.integers(0, len(pool), size=(n_perm, len(idx)))].sum(axis=1)
                    n_coins += 1
                    per_coin_pos += int(dr.mean() > base)
                if n_tot == 0:
                    continue
                obs = np.concatenate(obs)
                mean = obs.mean()
                p = (np.sum(rand_sum / n_tot >= mean) + 1) / (n_perm + 1)
                rows.append(dict(period=period, pattern=name, direction=direction, horizon=h,
                                 n=n_tot, n_coins=n_coins, n_coins_pos=per_coin_pos,
                                 hit_rate=(obs > 0).mean(), mean_ret_pct=100 * mean,
                                 baseline_pct=100 * base_sum / n_tot,
                                 excess_pct=100 * (mean - base_sum / n_tot), p_value=p))
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    # False-discovery-rate adjusted p-values (q-values)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def summarize_pooled(res: pd.DataFrame, horizon: int, min_n: int = 30) -> pd.DataFrame:
    # Side-by-side: earlier period vs recent. q_value adjusts the recent p-values for the
    # number of patterns tested, so 'edge holds' is not just the luckiest of ~50 tries.
    r = res[res['horizon'] == horizon]
    cols = ['n', 'n_coins_pos', 'hit_rate', 'excess_pct', 'p_value']
    ins = r[r['period'] == 'in_sample'].set_index('pattern')[cols].add_prefix('is_')
    rec = r[r['period'] == 'recent'].set_index('pattern')[cols].add_prefix('recent_')
    s = ins.join(rec, how='outer')
    s[['is_n', 'recent_n']] = s[['is_n', 'recent_n']].fillna(0).astype(int)
    ok = (s['is_n'] >= min_n) & (s['recent_n'] >= min_n)
    s['recent_q'] = np.nan
    s.loc[ok, 'recent_q'] = benjamini_hochberg(s.loc[ok, 'recent_p_value'].to_numpy())

    def verdict(row):
        if row['is_n'] < min_n or row['recent_n'] < min_n:
            return 'too few'
        both = row['is_excess_pct'] > 0 and row['recent_excess_pct'] > 0
        if both and row['recent_q'] < 0.1:
            return 'edge holds'
        if both and row['recent_p_value'] < 0.1:
            return 'promising'
        if both:
            return 'weak'
        if row['is_excess_pct'] > 0 >= row['recent_excess_pct']:
            return 'faded'
        return 'none'

    s['verdict'] = s.apply(verdict, axis=1)
    rank = {'edge holds': 0, 'promising': 1, 'weak': 2, 'faded': 3, 'none': 4, 'too few': 5}
    s['_o'] = s['verdict'].map(rank)
    return s.sort_values(['_o', 'recent_excess_pct'], ascending=[True, False]).drop(columns='_o')


def per_coin_table(coins: dict, pattern: str, direction: int, horizon: int) -> pd.DataFrame:
    # Excess return of one pattern on each coin, both periods
    rows = []
    for sym, c in coins.items():
        g = c['events']
        g = g[(g['pattern'] == pattern) & (g['direction'] == direction)]
        r = c['fwd'][horizon]
        row = {'coin': sym}
        for period, mask in (('in_sample', ~c['recent']), ('recent', c['recent'])):
            idx = g['i'].to_numpy()
            idx = idx[mask[idx]]
            idx = idx[~np.isnan(r[idx])]
            pool = np.flatnonzero(mask & ~np.isnan(r))
            row[f'{period}_n'] = len(idx)
            row[f'{period}_excess_pct'] = 100 * direction * (r[idx].mean() - r[pool].mean()) if len(idx) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('coin')


def active_signals(coins: dict, summary: pd.DataFrame, lookback: int, verdicts=('edge holds', 'promising')) -> pd.DataFrame:
    # Signals on each coin's last `lookback` bars, for patterns whose pooled verdict is in `verdicts`
    keep = set(summary.index[summary['verdict'].isin(verdicts)])
    rows = []
    for sym, c in coins.items():
        ev = c['events']
        ev = ev[(ev['i'] >= len(c['ohlc']) - lookback) & ev['pattern'].isin(keep)]
        for _, e in ev.iterrows():
            rows.append(dict(coin=sym, date=c['ohlc'].index[e['i']], pattern=e['pattern'],
                             direction='UP' if e['direction'] > 0 else 'DOWN',
                             price=c['ohlc']['close'].iloc[e['i']],
                             now=c['ohlc']['close'].iloc[-1],
                             verdict=summary.loc[e['pattern'], 'verdict']))
    return pd.DataFrame(rows).sort_values('date') if rows else pd.DataFrame(
        columns=['coin', 'date', 'pattern', 'direction', 'price', 'now', 'verdict'])
