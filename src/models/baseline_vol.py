"""
baseline_vol.py

Volatility task: baselines that the volatility LSTM has to beat, and the final comparison table.

Target (per forecast day t): target_logvol5 = log of the realized volatility over the NEXT 5 trading days,
    0.5 * log(mean of the next 5 daily squared log returns).
To show it in familiar units: annualized % = exp(value) * sqrt(252) * 100.

Input : data/processed/sequences_<TICKER>.npz        (from build_sequences.py, already purged and leakage-checked)
        data/processed/features.csv                  (daily returns and trailing volatility)
        data/predictions/lstm_vol_predictions.csv    (optional, from train_vol.py: adds the LSTM ensemble to the table)
Output: data/predictions/baseline_vol_metrics.csv

Baselines (per ticker, never mixing tickers). Everything is fitted on the TRAIN split only.
  const        : average log-volatility of the training period.
  persist_adj  : log(trailing 20-day volatility) + a constant offset. Slope fixed at 1 ("volatility stays as it was").
  persist_fit  : a + b * log(trailing 20-day volatility), a and b fitted. Lets the slope be below 1 (mean reversion).
  ewma_fit     : a + b * log(EWMA volatility, lambda 0.94, the RiskMetrics standard), a and b fitted.
  har          : HAR-style regression on three log-volatility components: last 1 day, last 5 days, last 20 days.
                 The standard serious benchmark for volatility forecasting. Daily squared returns stand in for
                 intraday realized variance, so it is a simplified HAR.
  ridge        : Ridge on the last day's 12 scaled features. Alpha chosen on VALIDATION.
None of these has a setting that is tuned on TEST. TEST is evaluated once.

Metrics (all on the log-volatility scale):
  mae_log, rmse_log : errors. Smaller is better.
  r2_oos            : 1 - SSE(model) / SSE(const). Measured against the TRAINING average, so it also rewards
                      noticing that the overall volatility LEVEL moved. See bias.
  corr              : correlation between forecast and outcome. Measures day-to-day tracking, ignores the level.
  bias              : mean(forecast - outcome). A large value means the model got the level wrong
                      (a model can have a good corr and still a bad rmse when the level shifted).
  rmse_vs_persist   : model RMSE / persist_adj RMSE. (Kept for continuity with the earlier tables.)
  rmse_vs_har       : model RMSE / har RMSE. Below 1.0 = better than the standard benchmark.

Caution: consecutive 5-day targets overlap by 4 days, so errors on neighbouring days are strongly correlated.
The effective sample size is roughly 1/5 of n. Treat differences of a few percent as noise.
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
LSTM_PRED_FILE = PROJECT_ROOT / "data" / "predictions" / "lstm_vol_predictions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "predictions" / "baseline_vol_metrics.csv"
FROZEN_FILE = PROJECT_ROOT / "models" / "vol_baselines.json"  # coefficients frozen for daily inference (predict.py)

TICKERS = ["AAPL", "MSFT", "NVDA"]
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
EVAL_SPLITS = ["val", "test"]
ANNUALIZE = np.sqrt(252)
EWMA_LAMBDA = 0.94
SQ_FLOOR = 1e-6  # added to a single day's squared return inside the log, so a near-zero day cannot give -inf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Data loading (shared with train_vol.py)
# --------------------------------------------------

def load_vol20_lookup() -> pd.DataFrame:
    """
    Volatility inputs for every ticker and date, indexed by (Ticker, 'YYYY-MM-DD').
    Every column at date t uses returns up to and including t only.
      vol_20   : trailing 20-day standard deviation of daily returns
      log_ewma : log of the EWMA volatility (RiskMetrics, lambda 0.94), started from vol_20 on the first row
      h1/h5/h20: log volatility over the last 1 / 5 / 20 days (the HAR components)
    """
    f = pd.read_csv(FEATURES_FILE, usecols=["Ticker", "Date", "ret_1", "vol_20"], parse_dates=["Date"])
    parts = []
    for ticker, g in f.groupby("Ticker", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        r = g["ret_1"].to_numpy(dtype=np.float64)
        s2 = np.empty(len(r))
        s2[0] = g["vol_20"].iloc[0] ** 2
        for t in range(1, len(r)):
            s2[t] = EWMA_LAMBDA * s2[t - 1] + (1 - EWMA_LAMBDA) * r[t] ** 2
        sq = pd.Series(r ** 2)
        parts.append(pd.DataFrame({
            "Ticker": ticker,
            "date_str": g["Date"].dt.strftime("%Y-%m-%d"),
            "vol_20": g["vol_20"],
            "log_ewma": 0.5 * np.log(s2),
            "h1": 0.5 * np.log(r ** 2 + SQ_FLOOR),
            "h5": 0.5 * np.log(sq.rolling(5).mean() + 1e-8),
            "h20": 0.5 * np.log(sq.rolling(20).mean() + 1e-8),
        }))
    return pd.concat(parts).set_index(["Ticker", "date_str"]).sort_index()


def load_ticker(ticker: str, lookup: pd.DataFrame) -> dict:
    """Usable volatility samples for each split: windows X, target y, dates and the baseline inputs."""
    path = SEQ_DIR / f"sequences_{ticker}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_features.py and build_sequences.py first.")
    d = np.load(path)
    if "volok_train" not in d.files:
        raise ValueError(f"{path.name} has no volatility arrays. Re-run build_features.py then build_sequences.py.")

    data = {}
    for s in ["train", "val", "test"]:
        ok = d[f"volok_{s}"]
        dates = d[f"dates_{s}"][ok]
        lk = lookup.loc[ticker].reindex(dates.tolist())
        cols = lk[["vol_20", "log_ewma", "h1", "h5", "h20"]].to_numpy(dtype=np.float64)
        assert np.isfinite(cols).all() and (lk["vol_20"].to_numpy() > 0).all(), \
            f"{ticker} {s}: bad trailing volatility values"
        data[s] = {
            "X": d[f"X_{s}"][ok],
            "y": d[f"yv_{s}"][ok].astype(np.float64),
            "dates": dates,
            "log_vol20": np.log(lk["vol_20"].to_numpy(dtype=np.float64)),
            "log_ewma": lk["log_ewma"].to_numpy(dtype=np.float64),
            "har": lk[["h1", "h5", "h20"]].to_numpy(dtype=np.float64),
        }
    return data


# --------------------------------------------------
# 3. Baselines and metrics
# --------------------------------------------------

def _fit_line(x: np.ndarray, y: np.ndarray):
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def baseline_predictions(data: dict) -> dict:
    """Return {split: {model_name: predictions}} for val and test, fitted on train only."""
    tr = data["train"]
    const = float(tr["y"].mean())
    offset = float(np.mean(tr["y"] - tr["log_vol20"]))
    a_p, b_p = _fit_line(tr["log_vol20"], tr["y"])
    a_e, b_e = _fit_line(tr["log_ewma"], tr["y"])
    har = LinearRegression().fit(tr["har"], tr["y"])

    X_tr = tr["X"][:, -1, :]
    best_alpha, best_rmse = None, np.inf
    for alpha in RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(X_tr, tr["y"])
        rmse = np.sqrt(np.mean((m.predict(data["val"]["X"][:, -1, :]) - data["val"]["y"]) ** 2))
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    ridge = Ridge(alpha=best_alpha).fit(X_tr, tr["y"])

    out = {"_meta": {
        "ridge_alpha": best_alpha, "const": const, "offset": offset,
        "persist_fit": (a_p, b_p), "ewma_fit": (a_e, b_e),
        "har_coef": har.coef_.tolist(), "har_intercept": float(har.intercept_),
    }}
    for s in EVAL_SPLITS:
        n = len(data[s]["y"])
        out[s] = {
            "const": np.full(n, const),
            "persist_adj": data[s]["log_vol20"] + offset,
            "persist_fit": a_p + b_p * data[s]["log_vol20"],
            "ewma_fit": a_e + b_e * data[s]["log_ewma"],
            "har": har.predict(data[s]["har"]),
            "ridge": ridge.predict(data[s]["X"][:, -1, :]),
        }
    return out


def vol_metrics(y: np.ndarray, pred: np.ndarray, const_value: float) -> dict:
    err = pred - y
    sse = float(np.sum(err ** 2))
    sse_const = float(np.sum((const_value - y) ** 2))
    corr = float("nan") if np.std(pred) == 0 else float(np.corrcoef(pred, y)[0, 1])
    return {
        "n": int(len(y)),
        "mae_log": float(np.mean(np.abs(err))),
        "rmse_log": float(np.sqrt(np.mean(err ** 2))),
        "r2_oos": 1.0 - sse / sse_const,
        "corr": corr,
        "bias": float(np.mean(err)),
    }


def _lstm_ensemble(lstm_df, ticker: str, split: str, data: dict):
    """Saved LSTM ensemble forecasts for this ticker and split, or None if they do not match the current data."""
    if lstm_df is None:
        return None
    sub = lstm_df[(lstm_df["ticker"] == ticker) & (lstm_df["split"] == split)]
    dates = data[split]["dates"].astype(str)
    if len(sub) != len(dates) or not np.array_equal(sub["forecast_date"].astype(str).to_numpy(), dates) \
            or not np.allclose(sub["actual_logvol"].to_numpy(), data[split]["y"], atol=1e-4):
        log.warning("%s %s: saved LSTM forecasts do not match the current data. Re-run train_vol.py to include them.",
                    ticker, split)
        return None
    return sub["predicted_logvol"].to_numpy(dtype=np.float64)


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    lookup = load_vol20_lookup()
    lstm_df = pd.read_csv(LSTM_PRED_FILE) if LSTM_PRED_FILE.exists() else None
    if lstm_df is None:
        log.info("No lstm_vol_predictions.csv found: showing baselines only (run train_vol.py to add the LSTM).")

    rows, coef_rows, frozen = [], [], {}
    for ticker in TICKERS:
        data = load_ticker(ticker, lookup)
        preds = baseline_predictions(data)
        meta = preds["_meta"]
        log.info("%s: ridge alpha %s | train mean log-vol %.3f (about %.1f%% annualized)",
                 ticker, meta["ridge_alpha"], meta["const"], np.exp(meta["const"]) * ANNUALIZE * 100)
        coef_rows.append({
            "ticker": ticker,
            "persist_fit_slope": meta["persist_fit"][1],
            "ewma_fit_slope": meta["ewma_fit"][1],
            "har_1d": meta["har_coef"][0], "har_5d": meta["har_coef"][1], "har_20d": meta["har_coef"][2],
            "har_sum": sum(meta["har_coef"]),
        })
        frozen[ticker] = {
            "const": meta["const"],
            "har_intercept": meta["har_intercept"],
            "har_coef": meta["har_coef"],
            "n_train": int(len(data["train"]["y"])),
            "fit_end_date": str(data["train"]["dates"][-1]),
        }

        for s in EVAL_SPLITS:
            y = data[s]["y"]
            all_preds = dict(preds[s])
            ens = _lstm_ensemble(lstm_df, ticker, s, data)
            if ens is not None:
                all_preds["lstm_ensemble"] = ens
            res = {name: vol_metrics(y, p, meta["const"]) for name, p in all_preds.items()}
            ref_p, ref_h = res["persist_adj"]["rmse_log"], res["har"]["rmse_log"]
            for name, r in res.items():
                rows.append({"ticker": ticker, "model": name, "split": s, **r,
                             "rmse_vs_persist": r["rmse_log"] / ref_p, "rmse_vs_har": r["rmse_log"] / ref_h})

    metrics = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved volatility comparison to %s", OUTPUT_FILE)

    FROZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_FILE.write_text(json.dumps({"har_components": ["h1", "h5", "h20"], "tickers": frozen}, indent=2))
    log.info("Saved frozen baseline coefficients to %s", FROZEN_FILE)

    print("\nFitted on TRAIN only (slopes below 1 = volatility mean-reverts; har_sum is the overall persistence):")
    print(pd.DataFrame(coef_rows).round(3).to_string(index=False))

    order = {"const": 0, "persist_adj": 1, "persist_fit": 2, "ewma_fit": 3, "har": 4, "ridge": 5, "lstm_ensemble": 6}
    show = ["ticker", "model", "mae_log", "rmse_log", "r2_oos", "corr", "bias", "rmse_vs_persist", "rmse_vs_har"]
    for s in EVAL_SPLITS:
        t = metrics[metrics["split"] == s].copy()
        t["_o"] = t["model"].map(order)
        t = t.sort_values(["ticker", "_o"])[show]
        print(f"\n===== {s.upper()} (log realized volatility, next 5 days) =====")
        print(t.round(4).to_string(index=False))

    print("\nHow to read this:")
    print("  rmse_vs_har < 1.0 means better than the standard HAR benchmark. This is the number that matters most.")
    print("  corr is day-to-day tracking; bias is the level error. A model can track well and still be off in level.")
    print("  Neighbouring targets overlap by 4 days, so the effective sample is about n/5: a few percent is noise.")


if __name__ == "__main__":
    main()
