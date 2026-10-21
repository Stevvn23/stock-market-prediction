"""
baseline.py

Phase 2, step 2: baseline models that the LSTM must beat.

Input : data/processed/features.csv   (from build_features.py)
Output: data/predictions/baseline_metrics.csv

Baselines (all trained per ticker, never mixing tickers):
  1. zero_return : predicts return = 0, i.e. "tomorrow's close = today's close" (persistence).
  2. train_mean  : predicts the mean next-day return of the training period.
                   Its sign is almost always "up", so its directional accuracy is
                   essentially the share of up days.
  3. ridge       : Ridge regression on the engineered features.
                   Scaler and Ridge are fitted on the TRAIN split only; alpha is chosen
                   on the VALIDATION split. The TEST split is evaluated once, untouched.

Metrics:
  mae_price / rmse_price : error of the recovered price, predicted_close = close * exp(pred_return)
  mae_ret / rmse_ret     : error on the return itself (comparable across tickers)
  dir_acc                : share of days where the predicted direction is correct
  dir_ci95               : +/- half-width of an approximate 95% interval for dir_acc
  rmse_ret_vs_zero       : model RMSE / zero_return RMSE on returns. Below 1.0 = better than persistence.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "predictions" / "baseline_metrics.csv"

NON_FEATURE_COLUMNS = {"Date", "Ticker", "Close", "target_date", "target_close", "target_ret", "split"}
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
EVAL_SPLITS = ["val", "test"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Metrics
# --------------------------------------------------

def evaluate(df: pd.DataFrame, pred_ret: np.ndarray) -> dict:
    """Compute price-level and return-level errors plus directional accuracy."""
    close = df["Close"].to_numpy()
    actual_close = df["target_close"].to_numpy()
    actual_ret = df["target_ret"].to_numpy()

    pred_close = close * np.exp(pred_ret)
    err_price = pred_close - actual_close
    err_ret = pred_ret - actual_ret
    n = len(df)

    if np.all(pred_ret == 0):
        dir_acc, dir_ci = np.nan, np.nan  # persistence makes no directional call
    else:
        dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))
        dir_ci = float(1.96 * np.sqrt(dir_acc * (1 - dir_acc) / n))

    return {
        "n": n,
        "mae_price": float(np.mean(np.abs(err_price))),
        "rmse_price": float(np.sqrt(np.mean(err_price ** 2))),
        "mae_ret": float(np.mean(np.abs(err_ret))),
        "rmse_ret": float(np.sqrt(np.mean(err_ret ** 2))),
        "dir_acc": dir_acc,
        "dir_ci95": dir_ci,
    }


# --------------------------------------------------
# 3. Baselines for ONE ticker
# --------------------------------------------------

def run_ticker(ticker: str, g: pd.DataFrame, feature_cols: list) -> list:
    train = g[g["split"] == "train"]
    parts = {s: g[g["split"] == s] for s in EVAL_SPLITS}

    # Fit scaler on train only
    scaler = StandardScaler().fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    y_train = train["target_ret"].to_numpy()
    X = {s: scaler.transform(parts[s][feature_cols]) for s in EVAL_SPLITS}

    # Choose Ridge alpha on validation RMSE (return scale)
    best_alpha, best_rmse = None, np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        pred_val = model.predict(X["val"])
        rmse = np.sqrt(np.mean((pred_val - parts["val"]["target_ret"].to_numpy()) ** 2))
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
    log.info("%s: Ridge alpha chosen on validation = %s", ticker, best_alpha)

    train_mean = float(y_train.mean())

    rows = []
    for split in EVAL_SPLITS:
        df_s = parts[split]
        n = len(df_s)
        preds = {
            "zero_return": np.zeros(n),
            "train_mean": np.full(n, train_mean),
            "ridge": ridge.predict(X[split]),
        }
        results = {name: evaluate(df_s, p) for name, p in preds.items()}
        zero_rmse = results["zero_return"]["rmse_ret"]
        for name, res in results.items():
            rows.append({
                "ticker": ticker,
                "model": name,
                "split": split,
                **res,
                "rmse_ret_vs_zero": res["rmse_ret"] / zero_rmse,
            })
    return rows


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} not found. Run src/features/build_features.py first.")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["Date", "target_date"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    log.info("Using %d features: %s", len(feature_cols), feature_cols)

    all_rows = []
    for ticker, g in df.groupby("Ticker", sort=True):
        all_rows.extend(run_ticker(ticker, g.sort_values("Date"), feature_cols))

    metrics = pd.DataFrame(all_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved baseline metrics to %s", OUTPUT_FILE)

    model_order = {"zero_return": 0, "train_mean": 1, "ridge": 2}
    show_cols = ["ticker", "model", "n", "mae_price", "rmse_price", "mae_ret", "rmse_ret",
                 "dir_acc", "dir_ci95", "rmse_ret_vs_zero"]
    for split in EVAL_SPLITS:
        table = metrics[metrics["split"] == split].copy()
        table["_o"] = table["model"].map(model_order)
        table = table.sort_values(["ticker", "_o"])[show_cols]
        print(f"\n===== {split.upper()} =====")
        print(table.round(4).to_string(index=False))

    print("\nHow to read this: rmse_ret_vs_zero < 1.0 means better than 'tomorrow = today'.")
    print("dir_acc is only meaningful relative to its dir_ci95 range; with ~185 test days, ")
    print("anything within about +/-7 points of 50% is statistically indistinguishable from a coin flip.")


if __name__ == "__main__":
    main()
