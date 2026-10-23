# Stock Market Prediction with Python and Power BI

An end-to-end, automated pipeline that forecasts US stocks (AAPL, MSFT, NVDA), compares a deep learning model
(LSTM) honestly against simple baselines, and tracks every forecast live in a Power BI dashboard.

> **Educational project. Not financial advice.** Price data comes from Yahoo Finance through the `yfinance`
> library and is used here for learning only. Raw vendor prices are not republished in this repository.

## What it does

- Downloads daily prices, builds 12 scale-free features, and trains small LSTM models for two targets:
  the next-day return and the 5-day realized volatility.
- Compares every model against baselines (persistence, training mean, HAR volatility, constant).
- Runs unattended every weekday with GitHub Actions: fetch, features, forecast, score, publish.
- Publishes forecasts and results as CSV files on a separate `data` branch, which Power BI reads.

## Honest results so far

| Target | Finding |
|---|---|
| Next-day return | No usable edge. The LSTM is within noise of "tomorrow equals today". |
| 5-day volatility | The LSTM did not reliably beat the HAR baseline. One stock (NVDA) shows a small, tentative gain. |

Validation results looked better than test results for the volatility model, which is the usual sign of
optimism from early stopping. The real test is the **live track record**: only forecasts made before the target
day opens are scored. It needs 60 or more trading days before it supports any statistical claim, so the dashboard
labels it "accumulating" until then.

## How the pipeline avoids fooling itself

- Chronological train, validation and test split with shared dates for all tickers.
- Scalers and target statistics fitted on training data only. Overlapping targets are purged at split boundaries.
- Any column that starts with `target_` is never a feature. Leakage tests perturb the future and check nothing moves.
- Forecasts are append-only. A row is `is_live = True` only if it was built on a completed session and issued before
  the target session opens. Backfilled rows are marked `is_live = False` and are never scored.
- Models are frozen files. The daily run does not retrain anything.

## Architecture

```
GitHub Actions (weekday schedule, with retries)
  fetch_data.py --strict  ->  build_features.py  ->  predict.py  ->  score_live.py  ->  publish to `data` branch
                                                                                              |
                                                                                    Power BI dashboard
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/data/fetch_data.py` | Download prices, drop partial bars, check freshness and gaps |
| `src/features/build_features.py` | Features, targets, chronological split |
| `src/models/` | Sequences, baselines, LSTM training, daily forecast (`predict.py`), live scoring (`score_live.py`) |
| `models/` | Frozen model files used by the daily forecast |
| `config/universe.json` | The list of tickers |
| `.github/workflows/daily_forecast.yml` | The automation |
| `scripts/` | Helpers that restore and publish the data branch |

Generated data lives on the `data` branch, not here: `features.csv`, `stock_predictions.csv`,
`pipeline_logs.csv`, `live_scores.csv`, `live_summary.csv`.

## Run it locally

Python 3.11 (64-bit). From the project root:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe src\data\fetch_data.py --strict
.\.venv\Scripts\python.exe src\features\build_features.py
.\.venv\Scripts\python.exe src\models\predict.py
```

To retrain (done deliberately, not daily), run in order: `build_features`, `build_sequences`, `baseline`,
`baseline_vol`, `train`, `train_vol`.

## Status

Phase 1 (three stocks, one LSTM per stock) is running live. Phase 2 will add more companies and a pooled model,
added next to the current one so the live track record of the first version stays intact.
