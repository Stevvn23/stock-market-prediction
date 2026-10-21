"""
baseline_vol.py

Volatility task, step 1: baselines that the volatility LSTM must beat.

Target (per forecast day t): target_logvol5 = log of the realized volatility over the NEXT 5 trading days,
    0.5 * log(mean of the next 5 daily squared log returns).
To show it in familiar units: annualized % = exp(value) * sqrt(252) * 100.

Input : data/processed/sequences_<TICKER>.npz   (from build_sequences.py, already purged and leakage-checked)
        data/processed/features.csv             (only used to look up the raw trailing 20-day volatility)
Output: data/predictions/baseline_vol_metrics.csv

Baselines (per ticker, never mixing tickers):
  1. const        : predicts the average log-volatility of the training period.
  2. persist_adj  : "volatility stays like the last 20 days". log(trailing 20-day volatility) plus a constant
                    offset estimated on TRAIN only (a raw trailing std is biased low versus a 5-day realized
                    measure, so without the offset the baseline would be handicapped unfairly).
  3. ridge        : Ridge regression on the last day's 12 scaled features. Alpha is chosen on VALIDATION.
                    Fitted on TRAIN only. TEST is evaluated once.

Metrics (all on the log-volatility scale):
  mae_log, rmse_log : errors. Smaller is better.
  r2_oos            : 1 - SSE(model) / SSE(const). Share of variance explained relative to the training average.
  corr              : Pearson correlation between forecast and outcome.
  rmse_vs_persist   : model RMSE / persist_adj RMSE. Below 1.0 = better than the persistence baseline.

Caution: consecutive 5-day targets overlap by 4 days, so errors on neighbouring days are strongly correlated.
The effective sample size is roughly 1/5 of n. Treat differences of a few percent as noise.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "predictions" / "baseline_vol_metrics.csv"

TICKERS = ["AAPL", "MSFT", "NVDA"]
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
EVAL_SPLITS = ["val", "test"]
ANNUALIZE = np.sqrt(252)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Data loading (shared with train_vol.py)
# --------------------------------------------------

def load_vol20_lookup() -> pd.Series:
    """Raw trailing 20-day volatility, indexed by (Ticker, 'YYYY-MM-DD')."""
    f = pd.read_csv(FEATURES_FILE, usecols=["Ticker", "Date", "vol_20"], parse_dates=["Date"])
    f["date_str"] = f["Date"].dt.strftime("%Y-%m-%d")
    return f.set_index(["Ticker", "date_str"])["vol_20"].sort_index()


def load_ticker(ticker: str, vol20_lookup: pd.Series) -> dict:
    """Usable volatility samples for each split: windows X, target y, dates, log trailing volatility."""
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
        vol20 = vol20_lookup.loc[ticker].reindex(dates.tolist()).to_numpy(dtype=np.float64)
        assert np.isfinite(vol20).all() and (vol20 > 0).all(), f"{ticker} {s}: bad trailing volatility values"
        data[s] = {
            "X": d[f"X_{s}"][ok],
            "y": d[f"yv_{s}"][ok].astype(np.float64),
            "dates": dates,
            "log_vol20": np.log(vol20),
        }
    return data


# --------------------------------------------------
# 3. Baselines and metrics
# --------------------------------------------------

def baseline_predictions(data: dict) -> dict:
    """Return {split: {model_name: predictions}} for val and test, fitted on train only."""
    tr = data["train"]
    const = float(tr["y"].mean())
    offset = float(np.mean(tr["y"] - tr["log_vol20"]))

    X_tr = tr["X"][:, -1, :]
    best_alpha, best_rmse = None, np.inf
    for alpha in RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(X_tr, tr["y"])
        rmse = np.sqrt(np.mean((m.predict(data["val"]["X"][:, -1, :]) - data["val"]["y"]) ** 2))
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    ridge = Ridge(alpha=best_alpha).fit(X_tr, tr["y"])

    out = {"_meta": {"ridge_alpha": best_alpha, "const": const, "offset": offset}}
    for s in EVAL_SPLITS:
        n = len(data[s]["y"])
        out[s] = {
            "const": np.full(n, const),
            "persist_adj": data[s]["log_vol20"] + offset,
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
    }


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    lookup = load_vol20_lookup()
    rows = []
    for ticker in TICKERS:
        data = load_ticker(ticker, lookup)
        preds = baseline_predictions(data)
        meta = preds["_meta"]
        log.info("%s: ridge alpha chosen on validation = %s | train mean log-vol = %.3f (about %.1f%% annualized)",
                 ticker, meta["ridge_alpha"], meta["const"], np.exp(meta["const"]) * ANNUALIZE * 100)
        for s in EVAL_SPLITS:
            y = data[s]["y"]
            res = {name: vol_metrics(y, p, meta["const"]) for name, p in preds[s].items()}
            ref = res["persist_adj"]["rmse_log"]
            for name, r in res.items():
                rows.append({"ticker": ticker, "model": name, "split": s, **r, "rmse_vs_persist": r["rmse_log"] / ref})

    metrics = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved baseline volatility metrics to %s", OUTPUT_FILE)

    order = {"const": 0, "persist_adj": 1, "ridge": 2}
    show = ["ticker", "model", "n", "mae_log", "rmse_log", "r2_oos", "corr", "rmse_vs_persist"]
    for s in EVAL_SPLITS:
        t = metrics[metrics["split"] == s].copy()
        t["_o"] = t["model"].map(order)
        t = t.sort_values(["ticker", "_o"])[show]
        print(f"\n===== {s.upper()} (log realized volatility, next 5 days) =====")
        print(t.round(4).to_string(index=False))

    print("\nHow to read this: rmse_vs_persist < 1.0 means better than 'volatility stays like the last 20 days'.")
    print("r2_oos is measured against the training average (const). Neighbouring targets overlap by 4 days,")
    print("so the effective sample is about n/5. Differences of a few percent are noise.")


if __name__ == "__main__":
    main()
