# Stock Market Prediction: Project Status (2026-10-23, day 3, GitHub Actions written)

Handoff note for continuing, with me or any other assistant. Replaces the 2026-10-22 version.

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
   checks pass, `--strict` exits with code 3 (nothing saved) when data is behind the latest completed session OR a
   NYSE session is missing in the last 100 calendar days (gap check).
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

## Done so far
Data fetch (hardened, with gap check), features, sequences, return and volatility baselines, return and volatility
LSTMs, evaluation and diagnosis, predict.py (first backfill run done), repository pushed to GitHub.
Day 3 (written, awaiting the user's first push and manual run): `.github/workflows/daily_forecast.yml`,
`scripts/restore_data.sh`, `scripts/publish_data.sh`, `requirements-ci.txt`.

## Automation design (GitHub Actions)
- Triggers: cron 23:30 UTC Mon-Fri (same evening as the session) plus retries 02:30, 06:30, 10:30 UTC Tue-Sat;
  `workflow_dispatch` with an `allow_backfill` checkbox (development only). One run at a time (concurrency group).
- Steps: checkout, Python 3.11, install `requirements-ci.txt` (tensorflow-cpu 2.21, keras 3.15.1, pinned like the dev
  machine), restore prediction history from the `data` branch, fetch --strict, build_features, predict, publish.
- Exit code 3 (data not ready, or forecast not live) is turned into a green run with a warning annotation, so the
  next scheduled retry can succeed. Any other non-zero exit fails the run (red, GitHub emails the owner).
- Frozen models and JSONs are committed in `main/models/`. Generated CSVs live on the orphan `data` branch:
  raw prices, features, stock_predictions.csv, pipeline_logs.csv. `publish_data.sh` refuses to publish if the new
  prediction file would drop any previously published line (protects the track record).
- Tested locally against a bare git remote: first publish (creates the branch), no-change run, append, restore into a
  fresh clone, stale-copy refusal. Workflow passes actionlint. NOT yet run on GitHub.
- Power BI raw URL pattern: https://raw.githubusercontent.com/<owner>/<repo>/data/data/predictions/stock_predictions.csv
  (repository must be public for Power BI Service).

## Open issue to check first
On 2026-10-22 (about 10:30 ET) Yahoo returned AAPL rows up to 10-20 plus a partial 10-22 bar, and NO row for 2026-10-21.
`fetch_data.py --strict` correctly refused to save. Re-run it on a morning (MYT, after 06:00). If 10-21 is still
missing: try a different request form (start/end dates instead of period) or add a fallback source.
predict.py refuses to forecast across a gap inside the 60-day window, by design.

## Plan for the next days (one evening session each, assuming 1 to 2 hours)
- Every morning until automation runs (MYT after 06:00): `fetch_data.py --strict`, then check the result.
- Fri 10/23: GitHub Actions (written; user commits, pushes and runs the manual test). A push before Fri 23:30 UTC
  means the first live forecast (Friday close, target Monday) gets logged by the scheduled run.
- Sat 10/24: `score_live.py` (scores only is_live=True forecasts once their target date has passed, returns based).
  Move the ticker list into one config file and remove the hard-coded lists (fetch_data, baseline, baseline_vol,
  train, train_vol, predict) so expansion is mostly configuration.
- Sun 10/25: Power BI data model. Decide to make the repository public (private raw files are not readable by Power BI
  Service; the data is public market data plus forecasts). Load the CSVs, build relationships and measures.
- Mon 10/26 and Tue 10/27: Power BI pages: Market Overview, Prediction and Evaluation, Pipeline Monitoring,
  Live Track Record (LSTM vs baselines on live forecasts only).
- Wed 10/28 and Thu 10/29: basic README (the repository is public by then), architecture diagram, screenshots,
  a few automated tests, monthly retraining procedure. No LinkedIn work yet.
Note: the Live Track Record needs 60 or more trading days (about 3 months) before it supports any statistical claim.
Present it as "accumulating" until then.

## Phase 2 (after the plan above): more companies and a bigger model
Start Fri 10/30, about 5 to 6 evenings, done around 11/5 (estimate; depends on Yahoo limits and training time).
Order: (1) pick the universe, test Yahoo at that scale, lengthen history; (2) config-driven code, new sequences and
splits; (3) pooled LSTM plus baselines plus optional LightGBM; (4) pooled evaluation and honest write-up of results;
(5) plug into the daily pipeline and Power BI (ticker slicer, model-version comparison).
The new model is ADDED next to the old one under a new version name (for example lstm_pool_v2). It never replaces
lstm_v1, so the live track record of v1 stays intact and the two versions can be compared on the same live days.
- Choose a fixed large-cap list and record the selection date (a list of today's index members has survivorship bias;
  live forecasts are not affected by it).
- Lengthen history to 10 years or more (more market regimes). Splits move, so everything is retrained.
- One pooled LSTM trained on all tickers (return and volatility) instead of one per ticker. Same chronological split
  dates for all tickers. Features are already scale-free. Consider scaling targets by each ticker's volatility so
  high-volatility names do not dominate the loss.
- Keep the baselines (persistence, training mean, HAR, constant). Optionally add a gradient-boosted tree model
  (LightGBM) as an extra benchmark.
- Pooled evaluation across tickers, with uncertainty that respects same-day correlation (block or date-clustered).
- Check Yahoo rate limits, GitHub Actions runtime, and storage size before scaling.

## Last step: LinkedIn and portfolio write-up (only after Phase 2)
The user wants to build the whole system first, and bigger, before any LinkedIn work. At the very end: final README
polish, final screenshots, results summary with honest conclusions, and the LinkedIn post.

## Git
Remote: origin = https://github.com/Stevvn23/stock-market-prediction.git (private), branch main, pushed.
Commit regularly: `git add src/ PROJECT_STATUS.md` then `git commit -m "..."` then `git push`.
