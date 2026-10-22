"""
predict.py

Daily forecast. For each ticker it records, side by side, what the LSTM models and the baselines
expect for (a) the next trading day's close and (b) the realized volatility of the next 5 trading days.

Input : data/processed/features.csv         (latest daily features; rebuilt every day by build_features.py)
        models/scalers.json                 (frozen at training time, never refitted here)
        models/lstm_meta.json, models/lstm_vol_meta.json, models/vol_baselines.json   (frozen at training time)
        models/lstm_<TICKER>_s<k>.keras, models/lstmvol_<TICKER>_s<k>.keras           (the seed ensembles)
Output: data/predictions/stock_predictions.csv   one row per ticker, forecast date, target type and model
        data/predictions/pipeline_logs.csv       one row per run (status, data dates, rows written, message)

Key rules
  * Frozen inputs. The scaler, target statistics and baseline coefficients come from the files written when the
    models were trained. This script never refits anything, so a new day of data cannot silently change the models.
    It checks that those files belong together and stops if they do not.
  * Live forecasts only. A forecast is written as is_live=True only if it is issued BEFORE the target session opens.
    If the data is stale (the vendor has not published the latest bar) or the target session has already started,
    nothing is written, the run is logged as not_live and the exit code is 3. --allow-backfill writes the rows
    anyway with is_live=False, for development only. Those rows never count as a track record.
  * Append only. A forecast that already exists (same ticker, forecast date, target type, model) is kept, not
    replaced, so re-running the job cannot rewrite history. --overwrite exists for development.
  * Dates come from the NYSE calendar (pandas-market-calendars), so weekends and holidays are skipped correctly.
    forecast_date is the last observed trading day; target_date is a different, later date.
  * The 60-day input window must be exactly the last 60 NYSE sessions. A gap in the data stops the run.

Models written per ticker and day
  close_1d : lstm_ens (seed average), persistence (tomorrow = today), train_mean (average training return)
  vol_5d   : lstm_ens (seed average), har (HAR regression), const (average training volatility)
For vol_5d, target_date is the LAST of the next 5 sessions. predicted_logvol is log of the daily realized
volatility over those 5 sessions; predicted_vol_ann_pct = exp(predicted_logvol) * sqrt(252) * 100.

Exit codes: 0 ok (including "nothing new to add"), 3 not live (stale data or target already open), 1 error.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # same numerics as training

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import keras

from baseline_vol import ANNUALIZE, load_vol20_lookup

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
PRED_FILE = PRED_DIR / "stock_predictions.csv"
LOG_FILE = PRED_DIR / "pipeline_logs.csv"

TICKERS = ["AAPL", "MSFT", "NVDA"]
VOL_HORIZON = 5
FEATURE_VERSION = "features_v1"

COLUMNS = ["ticker", "forecast_date", "target_type", "horizon_days", "target_date", "model", "model_version",
           "feature_version", "train_end_date", "is_live", "close_at_forecast", "predicted_ret", "predicted_close",
           "predicted_logvol", "predicted_vol_ann_pct", "run_utc"]
KEY = ["ticker", "forecast_date", "target_type", "model"]
LOG_COLUMNS = ["run_utc", "status", "forecast_dates", "rows_written", "message"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. NYSE calendar helpers
# --------------------------------------------------

def sessions_after(cal, date: pd.Timestamp, n: int) -> pd.DataFrame:
    """The next n NYSE sessions strictly after `date`, with market_open and market_close (UTC)."""
    sched = cal.schedule(start_date=date + pd.Timedelta(days=1), end_date=date + pd.Timedelta(days=40))
    if len(sched) < n:
        raise RuntimeError(f"NYSE calendar returned only {len(sched)} sessions after {date.date()}, need {n}.")
    return sched.iloc[:n]


def latest_completed_session(cal, now: pd.Timestamp) -> pd.Timestamp:
    """The most recent session whose market close is already in the past."""
    sched = cal.schedule(start_date=(now - pd.Timedelta(days=10)).tz_localize(None).normalize(),
                         end_date=now.tz_localize(None).normalize())
    done = sched[sched["market_close"] <= now]
    if done.empty:
        raise RuntimeError("Could not find a completed NYSE session in the last 10 days.")
    return done.index[-1]


def check_consecutive_sessions(cal, dates: pd.Series, ticker: str) -> None:
    """The model input must be exactly the last N NYSE sessions with no gaps and no extra days."""
    dates = pd.DatetimeIndex(dates)
    expected = cal.schedule(start_date=dates[0], end_date=dates[-1]).index
    if len(expected) != len(dates) or not (pd.DatetimeIndex(expected) == dates).all():
        missing = sorted(set(expected) - set(dates))
        extra = sorted(set(dates) - set(expected))
        raise RuntimeError(f"{ticker}: the last {len(dates)} rows are not consecutive NYSE sessions "
                           f"(missing {[d.date().isoformat() for d in missing][:5]}, "
                           f"unexpected {[d.date().isoformat() for d in extra][:5]}).")


# --------------------------------------------------
# 3. Model helpers
# --------------------------------------------------

def predict_seeds(prefix: str, ticker: str, n_seeds: int, X: np.ndarray) -> np.ndarray:
    """One standardized prediction per seed model."""
    out = []
    for k in range(n_seeds):
        path = MODEL_DIR / f"{prefix}_{ticker}_s{k}.keras"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train the models first (train.py / train_vol.py).")
        model = keras.saving.load_model(path, compile=False)
        out.append(float(model.predict(X, verbose=0).ravel()[0]))
        keras.backend.clear_session()
    return np.array(out)


def load_frozen():
    files = {"scalers": MODEL_DIR / "scalers.json", "ret_meta": MODEL_DIR / "lstm_meta.json",
             "vol_meta": MODEL_DIR / "lstm_vol_meta.json", "vol_base": MODEL_DIR / "vol_baselines.json"}
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run build_sequences.py, train.py, train_vol.py and baseline_vol.py first.")
    return {name: json.loads(p.read_text()) for name, p in files.items()}


def check_frozen_files_belong_together(ticker: str, fz: dict, feature_cols: list) -> None:
    """Scaler, return model, volatility model and baselines must all come from the same training run."""
    sc = fz["scalers"]
    window = sc["window"]
    fit_rows = sc["scalers"][ticker]["fit_rows"]
    rm, vm, vb = fz["ret_meta"][ticker], fz["vol_meta"][ticker], fz["vol_base"]["tickers"][ticker]
    problems = []
    if rm["window"] != window or vm["window"] != window:
        problems.append("window length differs between scalers and models")
    if rm["n_features"] != len(feature_cols) or vm["n_features"] != len(feature_cols):
        problems.append("number of features differs between scalers and models")
    if rm["train_samples"] != fit_rows - (window - 1):
        problems.append(f"return model trained on {rm['train_samples']} samples, scaler fitted on {fit_rows} rows")
    if vm["train_samples"] != fit_rows - (window - 1) - VOL_HORIZON:
        problems.append(f"volatility model trained on {vm['train_samples']} samples, scaler fitted on {fit_rows} rows")
    if vb["n_train"] != vm["train_samples"]:
        problems.append("HAR baseline and volatility LSTM were fitted on different training sets")
    if problems:
        raise RuntimeError(f"{ticker}: frozen files are out of sync ({'; '.join(problems)}). "
                           "Re-run build_sequences.py, train.py, train_vol.py and baseline_vol.py in that order.")


# --------------------------------------------------
# 4. Output helpers
# --------------------------------------------------

def make_row(**kw) -> dict:
    row = {c: np.nan for c in COLUMNS}
    row.update(kw)
    return row


def merge_and_save(new_rows: pd.DataFrame, overwrite: bool) -> int:
    """Append new rows; existing keys are kept unless overwrite. Atomic write. Returns rows added."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if PRED_FILE.exists():
        old = pd.read_csv(PRED_FILE, dtype={"forecast_date": str, "target_date": str, "train_end_date": str})
    else:
        old = pd.DataFrame(columns=COLUMNS)

    if overwrite and len(old):
        m = old.merge(new_rows[KEY], on=KEY, how="left", indicator=True)
        old = m[m["_merge"] == "left_only"].drop(columns="_merge")
        to_add = new_rows
    else:
        m = new_rows.merge(old[KEY], on=KEY, how="left", indicator=True)
        to_add = m[m["_merge"] == "left_only"].drop(columns="_merge")

    out = pd.concat([old, to_add], ignore_index=True) if len(old) else to_add.copy()
    out = out.sort_values(["forecast_date", "ticker", "target_type", "model"]).reset_index(drop=True)[COLUMNS]
    tmp = PRED_FILE.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, PRED_FILE)
    return len(to_add)


def write_log(run_utc: str, status: str, forecast_dates: str, rows_written: int, message: str) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"run_utc": run_utc, "status": status, "forecast_dates": forecast_dates,
                         "rows_written": rows_written, "message": message}])[LOG_COLUMNS]
    row.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)


# --------------------------------------------------
# 5. The run
# --------------------------------------------------

def run(now: pd.Timestamp, args):
    run_utc = now.strftime("%Y-%m-%d %H:%M:%S")
    cal = mcal.get_calendar("NYSE")
    fz = load_frozen()
    feature_cols = fz["scalers"]["feature_columns"]
    window = fz["scalers"]["window"]

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} not found. Run build_features.py first.")
    feat = pd.read_csv(FEATURES_FILE, parse_dates=["Date"])
    missing_cols = [c for c in feature_cols if c not in feat.columns]
    if missing_cols:
        raise RuntimeError(f"features.csv lacks columns the models were trained on: {missing_cols}")

    vol_lookup = load_vol20_lookup()
    latest_done = latest_completed_session(cal, now)

    rows, notes, fdates = [], [], []
    for ticker in TICKERS:
        check_frozen_files_belong_together(ticker, fz, feature_cols)
        g = feat[feat["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(g) < window:
            raise RuntimeError(f"{ticker}: only {len(g)} rows in features.csv, need {window}.")
        last = g.iloc[-1]
        fdate = pd.Timestamp(last["Date"])
        fdates.append(f"{ticker}:{fdate.date().isoformat()}")

        win = g.iloc[-window:]
        check_consecutive_sessions(cal, win["Date"], ticker)
        F = win[feature_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(F).all():
            raise RuntimeError(f"{ticker}: NaN or inf in the latest feature window.")

        nxt = sessions_after(cal, fdate, VOL_HORIZON)
        t1, t5 = nxt.index[0], nxt.index[-1]
        is_live = bool(nxt.iloc[0]["market_open"] > now)
        if not is_live:
            if fdate < latest_done:
                why = (f"stale data (last bar {fdate.date()}, latest completed session {latest_done.date()}); "
                       "the vendor may not have published the latest bar yet")
            else:
                why = f"the target session {t1.date()} has already opened"
            notes.append(f"{ticker}: {why}")
            if not args.allow_backfill:
                continue

        st = fz["scalers"]["scalers"][ticker]
        X = ((F - np.array(st["mean"])) / np.array(st["scale"])).astype(np.float32)[None, :, :]
        close = float(last["Close"])
        train_end = st["fit_end_date"]
        common = dict(ticker=ticker, forecast_date=fdate.date().isoformat(), is_live=is_live, close_at_forecast=close,
                      feature_version=FEATURE_VERSION, train_end_date=train_end, run_utc=run_utc)

        # ---- next-day close ----
        rm = fz["ret_meta"][ticker]
        seed_ret = predict_seeds("lstm", ticker, rm["n_seeds"], X) * rm["y_std"] + rm["y_mean"]
        ret_models = {"lstm_ens": (float(seed_ret.mean()), rm["model_version"]),
                      "persistence": (0.0, "baseline"),
                      "train_mean": (float(rm["y_mean"]), "baseline")}
        for name, (r, ver) in ret_models.items():
            rows.append(make_row(**common, target_type="close_1d", horizon_days=1, target_date=t1.date().isoformat(),
                                 model=name, model_version=ver, predicted_ret=r, predicted_close=close * float(np.exp(r))))

        # ---- next-5-day volatility ----
        vm = fz["vol_meta"][ticker]
        vb = fz["vol_base"]["tickers"][ticker]
        seed_vol = predict_seeds("lstmvol", ticker, vm["n_seeds"], X) * vm["y_std"] + vm["y_mean"]
        comp = vol_lookup.loc[(ticker, fdate.strftime("%Y-%m-%d"))]
        har = vb["har_intercept"] + float(np.dot(vb["har_coef"], comp[["h1", "h5", "h20"]].to_numpy(dtype=np.float64)))
        vol_models = {"lstm_ens": (float(seed_vol.mean()), vm["model_version"]),
                      "har": (float(har), "baseline"),
                      "const": (float(vb["const"]), "baseline")}
        for name, (v, ver) in vol_models.items():
            rows.append(make_row(**common, target_type="vol_5d", horizon_days=VOL_HORIZON, target_date=t5.date().isoformat(),
                                 model=name, model_version=ver, predicted_logvol=v,
                                 predicted_vol_ann_pct=float(np.exp(v) * ANNUALIZE * 100)))

    forecast_dates = ";".join(fdates)
    if not rows:
        return "not_live", "No forecast written. " + " | ".join(notes), 0, forecast_dates, pd.DataFrame()

    new_rows = pd.DataFrame(rows)[COLUMNS]
    added = merge_and_save(new_rows, args.overwrite)
    live_all = bool(new_rows["is_live"].all())
    if added == 0:
        status, msg = "no_new_rows", "All forecasts for these dates already exist; nothing added."
    elif live_all:
        status, msg = "ok", f"{added} forecast rows added."
    else:
        status, msg = "backfill", f"{added} rows added with is_live=False. " + " | ".join(notes)
    return status, msg, added, forecast_dates, new_rows


def print_summary(new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return
    print("\nNext trading day (close_1d):")
    r = new_rows[new_rows["target_type"] == "close_1d"]
    piv = r.pivot(index="ticker", columns="model", values="predicted_ret") * 100
    info = r[r["model"] == "lstm_ens"].set_index("ticker")[["forecast_date", "target_date", "close_at_forecast", "predicted_close"]]
    print(pd.concat([info, piv.add_prefix("ret% ")], axis=1).round(3).to_string())
    print("\nNext 5 trading days, annualized realized volatility % (vol_5d):")
    v = new_rows[new_rows["target_type"] == "vol_5d"]
    pv = v.pivot(index="ticker", columns="model", values="predicted_vol_ann_pct")
    tinfo = v[v["model"] == "lstm_ens"].set_index("ticker")[["target_date"]].rename(columns={"target_date": "window_ends"})
    print(pd.concat([tinfo, pv], axis=1).round(1).to_string())
    print("\nThe LSTM return forecasts are tiny on purpose: the evaluation showed no reliable edge, so they sit")
    print("close to the average return. They are recorded so the live track record can test that honestly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", default=None, help="UTC time to act as 'now', e.g. 2026-10-21T23:30:00Z (for tests)")
    parser.add_argument("--allow-backfill", action="store_true", help="development only: write rows even if not live (is_live=False)")
    parser.add_argument("--overwrite", action="store_true", help="development only: replace existing rows with the same key")
    args = parser.parse_args()

    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    run_utc = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        status, msg, added, fdates, new_rows = run(now, args)
    except Exception as exc:
        write_log(run_utc, "error", "", 0, f"{type(exc).__name__}: {exc}")
        log.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)

    write_log(run_utc, status, fdates, added, msg)
    log.info("status=%s | %s", status, msg)
    print_summary(new_rows)
    if status == "not_live":
        sys.exit(3)


if __name__ == "__main__":
    main()
