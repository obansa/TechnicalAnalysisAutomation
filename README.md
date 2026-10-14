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
