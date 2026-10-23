import itertools
import numpy as np
import pandas as pd

# Turns pattern signals into actual trades with costs.
#
# - Signal bar i: first bar where the pattern is fully known (from pattern_backtest.collect_events).
# - Entry: OPEN of bar i+1 (a signal on a daily close can't be traded at that same close).
# - Exit: stop-loss / take-profit measured in ATR (average true range) at the signal bar, or after
#   `hold` bars at the close, whichever comes first. If a bar touches both stop and target, the
#   stop is assumed to hit first (the pessimistic choice).
# - Costs: `cost` is charged on each side (fee + slippage), so a round trip costs 2 * cost.
# - One open position per coin and pattern: signals that arrive while a trade is open are skipped.
# - Shorts ignore funding/borrow costs.
#
# Exit rules are tuned on the earlier period only and then applied unchanged to the recent period.

DEFAULT_COST = 0.0015  # 0.10% exchange fee + 0.05% slippage, per side


def atr(ohlc: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = ohlc['high'], ohlc['low'], ohlc['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _walk(h, l, c, entry_i, last, direction, entry, stop, tgt):
    # Walk bars entry_i..last. Returns (exit_price, exit_index, reason) with reason 'stop', 'target' or 'time'.
    for j in range(entry_i, last + 1):
        if direction > 0:
            if stop is not None and l[j] <= stop:
                return stop, j, 'stop'
            if tgt is not None and h[j] >= tgt:
                return tgt, j, 'target'
        else:
            if stop is not None and h[j] >= stop:
                return stop, j, 'stop'
            if tgt is not None and l[j] <= tgt:
                return tgt, j, 'target'
    return c[last], last, 'time'


def _levels(entry, a, direction, stop_mult, tgt_mult):
    stop = None if stop_mult is None else entry - direction * stop_mult * a
    tgt = None if tgt_mult is None else entry + direction * tgt_mult * a
    return stop, tgt


def simulate_trade(o, h, l, c, a, i, direction, stop_mult, tgt_mult, hold, cost):
    # Returns (exit_index, net_log_return, gross_log_return) or None if the trade can't be run.
    entry_i = i + 1
    last = min(i + hold, len(c) - 1)
    if entry_i > last or np.isnan(a[i]) or a[i] <= 0:
        return None
    entry = o[entry_i]
    stop, tgt = _levels(entry, a[i], direction, stop_mult, tgt_mult)
    exit_price, exit_i, _ = _walk(h, l, c, entry_i, last, direction, entry, stop, tgt)
    gross = direction * np.log(exit_price / entry)
    return exit_i, gross - 2 * cost, gross


def simulate_trade_live(o, h, l, c, a, i, direction, stop_mult, tgt_mult, hold, cost):
    # Like simulate_trade, but aware that the data ends at the last bar.
    # Returns a dict with status 'PENDING' (enters at the next open, not yet known), 'OPEN'
    # (still inside its holding period, marked at the last close) or 'CLOSED' (reason stop/target/time).
    n = len(c)
    entry_i = i + 1
    if entry_i > n - 1:
        return dict(status='PENDING', entry_i=entry_i)
    if np.isnan(a[i]) or a[i] <= 0:
        return None
    entry = o[entry_i]
    stop, tgt = _levels(entry, a[i], direction, stop_mult, tgt_mult)
    last = min(i + hold, n - 1)
    price, exit_i, reason = _walk(h, l, c, entry_i, last, direction, entry, stop, tgt)
    gross = direction * np.log(price / entry)
    still_open = reason == 'time' and i + hold > n - 1
    return dict(status='OPEN' if still_open else 'CLOSED', reason='' if still_open else reason,
                entry_i=entry_i, exit_i=exit_i, entry=entry, stop=stop, tgt=tgt, price=price,
                gross=gross, net=gross - 2 * cost)


def run_trades(coins: dict, pattern: str, direction: int, stop_mult, tgt_mult, hold: int,
               cost: float = DEFAULT_COST, eligible: dict = None) -> pd.DataFrame:
    # coins: output of cross_coin.prepare (daily). Returns one row per trade.
    # eligible: optional {coin: bool array over bars}; signals on bars where it is False are ignored.
    rows = []
    for sym, cd in coins.items():
        ohlc = cd['ohlc']
        o, h, l, c = (ohlc[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
        a = cd.get('atr')
        if a is None:
            a = cd['atr'] = atr(ohlc)
        ev = cd['events']
        sig = np.sort(ev.loc[(ev['pattern'] == pattern) & (ev['direction'] == direction), 'i'].to_numpy())
        if eligible is not None:
            sig = sig[eligible[sym][sig]]
        busy_until = -1
        for i in sig:
            if i <= busy_until:
                continue
            res = simulate_trade(o, h, l, c, a, int(i), direction, stop_mult, tgt_mult, hold, cost)
            if res is None:
                continue
            exit_i, net, gross = res
            busy_until = exit_i
            rows.append((sym, ohlc.index[i + 1], ohlc.index[exit_i], net, gross))
    return pd.DataFrame(rows, columns=['coin', 'entry', 'exit', 'net', 'gross'])


def trade_stats(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return dict(n=0, win_rate=np.nan, avg_net_pct=np.nan, avg_gross_pct=np.nan,
                    profit_factor=np.nan, total_net_pct=np.nan, worst_pct=np.nan)
    wins, losses = t.loc[t['net'] > 0, 'net'].sum(), -t.loc[t['net'] <= 0, 'net'].sum()
    return dict(n=len(t), win_rate=(t['net'] > 0).mean(), avg_net_pct=100 * t['net'].mean(),
                avg_gross_pct=100 * t['gross'].mean(),
                profit_factor=wins / losses if losses > 0 else np.inf,
                total_net_pct=100 * t['net'].sum(), worst_pct=100 * t['net'].min())


GRID = dict(stop=[None, 1.0, 2.0, 3.0], tgt=[None, 2.0, 3.0, 4.0], hold=[5, 10, 20])


def tune_exit_rules(coins: dict, pattern: str, direction: int, test_start, cost=DEFAULT_COST,
                    min_trades: int = 30, grid=GRID) -> pd.DataFrame:
    # Tries every exit-rule combination on trades that ENTER before test_start only.
    test_start = pd.Timestamp(test_start)
    rows = []
    for stop, tgt, hold in itertools.product(grid['stop'], grid['tgt'], grid['hold']):
        t = run_trades(coins, pattern, direction, stop, tgt, hold, cost)
        t = t[t['entry'] < test_start]
        if len(t) < min_trades:
            continue
        rows.append(dict(stop=stop, tgt=tgt, hold=hold, **trade_stats(t)))
    return pd.DataFrame(rows).sort_values('avg_net_pct', ascending=False).reset_index(drop=True)


def random_entry_baseline(coins: dict, direction: int, stop_mult, tgt_mult, hold: int, test_start,
                          n_trades_per_coin: dict, cost=DEFAULT_COST, n_perm: int = 300, seed: int = 0,
                          eligible: dict = None, pool_end=None) -> np.ndarray:
    # Mean net return per trade when the same number of trades per coin are entered on random
    # days in the recent period, in the same direction with the same exit rules.
    # eligible: same filter as the pattern trades used, so a filter cannot create an edge by itself.
    rng = np.random.default_rng(seed)
    test_start = pd.Timestamp(test_start)
    prep = {}
    for sym, n in n_trades_per_coin.items():
        cd = coins[sym]
        ohlc = cd['ohlc']
        if 'atr' not in cd:
            cd['atr'] = atr(ohlc)
        first = int(np.searchsorted(ohlc.index, test_start)) - 1
        pool = np.arange(max(first, 14), len(ohlc) - 2)
        if pool_end is not None:   # only entries before pool_end (to benchmark an earlier period)
            pool = pool[ohlc.index[pool] < pd.Timestamp(pool_end)]
        if eligible is not None:
            pool = pool[eligible[sym][pool]]
        if len(pool) == 0:
            continue
        prep[sym] = (ohlc, pool, n)
    means = np.empty(n_perm)
    for p in range(n_perm):
        rets = []
        for sym, (ohlc, pool, n) in prep.items():
            o, h, l, c = (ohlc[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
            a = coins[sym]['atr']
            for i in rng.choice(pool, size=n):
                res = simulate_trade(o, h, l, c, a, int(i), direction, stop_mult, tgt_mult, hold, cost)
                if res is not None:
                    rets.append(res[1])
        means[p] = np.mean(rets) if rets else np.nan
    return means
