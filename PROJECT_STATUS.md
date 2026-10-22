# Stock Market Prediction: Project Status (2026-10-22, end of day 2)

Handoff note for continuing, with me or any other assistant. Replaces the 2026-10-21 version.

## Fixed requirements (unchanged)
US stocks AAPL, MSFT, NVDA. LSTM. Power BI. Daily automation (GitHub Actions). CSV storage. Free tools.
Windows PowerShell. Python 3.11 64-bit, project venv at `.venv`. TensorFlow 2.21, Keras 3.15, pandas 3.0.6.
Always call the venv interpreter explicitly: `.\.venv\Scripts\python.exe ...`.
Never create text files with PowerShell `>` (UTF-16 trap). Edit files in VS Code or copy the delivered files.
The user is in Malaysia (UTC+8). US close (16:00 ET) is 04:00 MYT while US daylight time lasts (until 2026-11-01), then 05:00 MYT.

## Pipeline (all scripts run from the project root)
1. `src/data/fetch_data.py` -> `data/raw/stock_prices.csv` (5y daily). Price definition: Close is SPLIT-adjusted but
   NOT dividend-adjusted (`auto_adjust=False`). Keep it that way everywhere. Dividend effect on these three stocks is
   small (an ex-dividend day drops the close by about 0.2 to 0.3 percent, against 1.5 to 3.5 percent daily volatility).
   Hardened version: drops a partial bar from a session still open, retries downloads, writes atomically only after all
   checks pass, `--strict` exits with code 3 (nothing saved) when data is behind the latest completed session.
2. `src/features/build_features.py` -> `data/processed/features.csv`
   12 scale-free features. Targets: `target_ret` (next-day log return), `target_logvol5` (log realized volatility,
   next 5 days). Any column starting with `target_` is never a feature. Split by percentage (70/15/15), shared dates.
3. `src/models/build_sequences.py` -> `data/processed/sequences_<T>.npz`, `models/scalers.json`
   60-day windows per ticker, scaler fitted on train only, volatility samples purged at split boundaries.
4. Return task: `baseline.py`, `train.py` (5 seeds per ticker, Huber loss, early stopping on val).
5. Volatility task: `baseline_vol.py` (const, persist_adj, persist_fit, ewma_fit, har, ridge, plus the saved LSTM
   ensemble; writes `models/vol_baselines.json`), `train_vol.py` (same structure as train.py).
6. `src/diagnostics/diagnose_vol.py`: diagnostic only, never used to tune.
7. `src/models/predict.py`: daily forecast, see below.

## Evaluation results (test split, ~185 trading days, one look per model version)
Return task: no usable edge. rmse_ret_vs_zero for the LSTM ensemble 0.994 / 1.008 / 1.001 (AAPL / MSFT / NVDA);
baselines the same within noise; direction accuracy within noise of 50%.

Volatility task (5-day realized vol, log scale), rmse_vs_har (below 1.0 = better than HAR):
- LSTM ensemble 1.006 / 0.998 / 0.879. const (training average) 0.958 / 0.988 / 1.139.
- AAPL and MSFT: nothing beats the training average on test. NVDA: LSTM best, but the gain is mostly level
  tracking (bias 0.09 vs 0.17 for HAR, 0.32 for const); its forecasts vary very little day to day, and corr is 0.15
  with an effective sample of about 36 (corr noise about +-0.17). Not enough to claim skill.
- On val the LSTM beat HAR by 4 to 9 percent on all three; that advantage vanished on AAPL and MSFT in test
  (early stopping on val is optimistic). Persistence with slope 1 overreacts (train slope is 0.43 to 0.58).
- Diagnosis: data is clean (largest moves are real events). The test period is calm and trailing volatility lost
  its predictive power there (corr about 0), so little is predictable.
README-ready conclusion: no edge on next-day returns; deep learning did not reliably beat HAR on 5-day volatility
(one stock shows a tentative gain); the live track record is the real test.

## predict.py design (tested on synthetic data, first real run done with --allow-backfill)
- Writes `data/predictions/stock_predictions.csv` (6 rows per ticker per day: close_1d for lstm_ens, persistence,
  train_mean; vol_5d for lstm_ens, har, const) and `data/predictions/pipeline_logs.csv`.
- is_live=True only if built on a COMPLETED session and issued before the target session opens. Otherwise nothing
  is written and exit code is 3. `--allow-backfill` writes rows with is_live=False (development only).
  Rows with is_live=False must never be counted in the track record.
- Append only (existing keys are kept). Frozen scalers, target stats and baseline coefficients. Consistency checks
  between the frozen files. NYSE calendar for target dates. Fails on a gap inside the 60-day window.
- DAILY PIPELINE = fetch -> build_features -> predict. Do NOT run build_sequences or any training script daily:
  the split dates move as data grows and would desynchronize scalers and models. Retrain (all scripts in order:
  build_features, build_sequences, baseline, baseline_vol, train, train_vol) only deliberately, e.g. monthly.

## Rules to keep
- Never tune or select anything using the test split. Tune on val only.
- Chronological splits, train-only scalers, separate sequences per ticker, forecast date separate from target date.
- Report results honestly. No claims of predictive accuracy without measured evidence.
- Treat test rmse_ret_vs_zero below 0.95 or volatility r2_oos above 0.7 as suspicious: check leakage.
- Close is split-adjusted, so history is re-scaled retroactively after a stock split (rare). Score live forecasts on
  returns computed from one consistent download, not on stored price levels.
- Known risk for automation: Yahoo (yfinance) sometimes rate-limits or blocks requests from cloud runners such as
  GitHub Actions. Retries are built in; if it proves flaky, add a fallback data source.

## Next steps (suggested order)
1. Create the GitHub repository (private first) and push. The work exists only on the user's computer. Local branch
   is master, no remote yet; rename to main.
2. Decide how models persist for GitHub Actions. `.gitignore` currently excludes `*.keras` and `models/*.json`,
   so a runner would have no models. Recommended: commit the small frozen model files and JSONs, and let the Action
   push the output CSVs to a separate `data` branch (free, and Power BI can read the raw CSV URLs).
3. `src/models/score_live.py`: for live forecasts whose target date has passed, compute actual return / 5-day
   volatility from raw prices and write `live_scores.csv` (only is_live=True rows).
4. Power BI: data model from the CSVs; pages Market Overview, Prediction and Evaluation, Pipeline Monitoring,
   plus a Live Track Record page (LSTM vs baselines on live forecasts only).
5. GitHub Actions: daily schedule after the close plus vendor delay (and before the next open), retries, holiday
   awareness, failure alerts. Then tests, README, screenshots.

## Git
Local repository on branch master, no remote. Commit regularly:
`git add .gitignore src/ requirements.txt requirements-lock.txt` then `git commit -m "..."`.
