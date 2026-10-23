# TechnicalAnalysisAutomation
Python functions for automating technical analysis. 

Install the dependencies:

    pip install numpy pandas scipy matplotlib mplfinance scikit-learn

The PIP pattern miner now uses scikit-learn instead of pyclustering, so the old numpy==1.23.1 pin is no longer needed.
(`mp_support_resist.py` and `retracement_ratios.py` also need `pandas_ta`.) 

## New patterns and prediction backtest

| File | What it detects |
|---|---|
| `double_top_bottom.py` | Double and triple tops/bottoms, confirmed on a close through the neckline |
| `triangles_wedges.py` | Ascending/descending/symmetric triangles and rising/falling wedges, reported at breakout |
| `divergences.py` | Regular and hidden RSI / MACD-histogram divergences, signalled when the second swing is confirmed |
| `triangles_wedges.py` (`PARALLEL_SHAPES`) | Rectangles, rising/falling channels and broadening (megaphone) formations |
| `candlestick_patterns.py` | Engulfing, hammer, hanging man, shooting star, inverted hammer, doji, morning/evening star, three soldiers/crows |

Every detector reports a pattern on the first bar where it is fully formed, so nothing uses future data.

### Get recent data
`data_fetcher.py` downloads candles from Binance's public API (no API key needed) and adds the newest ones to the existing CSVs:

    python data_fetcher.py --interval 1d     # updates BTCUSDT86400.csv
    python data_fetcher.py --interval 1h     # updates BTCUSDT3600.csv
    python data_fetcher.py --symbol ETHUSDT --interval 4h

### Test whether patterns predict
`pattern_backtest.py` runs every detector, including head & shoulders and flags/pennants. For each pattern it measures the forward return, adjusted for direction, against random entries, separately for an in-sample period and a recent period:

    python pattern_backtest.py --data BTCUSDT86400.csv --test-start 2024-01-01
    python pattern_backtest.py --data BTCUSDT3600.csv --test-start 2025-01-01 --horizon 24

The verdict column reads `edge holds` only if the pattern beat random entries in both periods and the recent result is significant (permutation p-value < `--alpha`). Because ~30 patterns are tested at 4 horizons, a few will look significant by chance, so trust only edges that hold across periods and horizons. Full results go to `pattern_results.csv`.

### Analysis notebook
`pattern_analysis.ipynb` runs the divergences, rectangles/channels/broadening, and the PIP pattern miner (trained before 2024, tested on 2024 onward) on the latest data, with charts and conclusions. Re-run it after updating the data:

    jupyter nbconvert --to notebook --execute --inplace pattern_analysis.ipynb

### Other coins
`cross_coin.py` runs every detector on ETH, SOL, BNB, XRP, ADA and DOGE (data in `data/`), pools the signals per pattern, and corrects for the number of patterns tested. `coin_analysis.ipynb` shows the results, checks whether the Bitcoin leads replicate, and lists currently active signals. Download the coin data with:

    python data_fetcher.py --symbol ETHUSDT --interval 1d --out data/ETHUSDT86400.csv

### Trading the patterns
`trade_sim.py` turns signals into trades: next-day-open entry, ATR-based stop and target, a maximum holding time, and 0.15% costs per side. Exit rules are tuned on data before 2024 only. `trade_analysis.ipynb` runs this for the five best daily patterns on the six altcoins, with a random-entry baseline, year-by-year and per-coin breakdowns, a cost sensitivity table and a Bitcoin check.

### Proving the strategies, and paper trading
`strategy_lab.py` adds trend/volume filters, 4-hour bars and combined signals, and applies a strict protocol: filter and exit rules are tuned on pre-2024 trades only, then compared with filter-matched random entries, with a multiple-testing correction. `proof_analysis.ipynb` shows the results (2 of 40 strategies passed; they are the same idea: shorting daily MACD bearish divergences).

`paper_trade.py` is a forward test. It never places orders; it logs what the rules in `paper_strategies.json` would have done on new daily candles since the start date in `paper_state.json`:

    python paper_trade.py            # download new candles, update paper_trades.csv, print the record

### New pattern detectors (branch `ophand`)
`smart_money.py` detects smart money concepts (break of structure, change of character, liquidity sweeps, fair value gaps, order blocks with retests) and `cup_handle.py` detects cup and handle. All report a pattern on the bar where it is first known. `tests/test_detectors.py` (run `python -m pytest tests`) checks each definition on hand-built candles, planted and malformed cups, and re-runs every detector on truncated data to prove there is no lookahead. `new_pattern_analysis.ipynb` shows the accuracy checks, then tests the patterns on six coins at daily, 4-hour and hourly bars (result: mostly a momentum effect, nothing proven as a trade).

### Analysis of a broker article's strategies
`vt_strategies.py` codes the rule-based strategies from the VT Markets article "10 Best Trading Strategies for 2026" (trend following, breakout, range/mean reversion) with the parameters it states, and `vt_strategies_analysis.ipynb` checks the article's claims and tests them on six coins with fees and random-entry comparisons. Result: 7 of 30 variants made money since 2024 and none passes the multiple-testing correction.
