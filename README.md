# TechnicalAnalysisAutomation
Python functions for automating technical analysis. 

Important: 
The pyclustering library does not appear to work with the current version of numpy.

pip install numpy==1.23.1

This numpy version works for me. 

## New patterns and prediction backtest

| File | What it detects |
|---|---|
| `double_top_bottom.py` | Double and triple tops/bottoms, confirmed on a close through the neckline |
| `triangles_wedges.py` | Ascending/descending/symmetric triangles and rising/falling wedges, reported at breakout |
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
