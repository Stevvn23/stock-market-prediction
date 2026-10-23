"""
train.py

Phase 3: train one small LSTM per ticker and compare it with the baselines.

Input : data/processed/sequences_<TICKER>.npz      (from build_sequences.py)
        data/predictions/baseline_metrics.csv      (optional, from baseline.py, for the final table)
Output: models/lstm_<TICKER>_s<SEED>.keras         one model per ticker and seed
        models/lstm_meta.json                      target scaling stats and training info (needed for inference)
        data/predictions/lstm_metrics.csv          metrics on val and test
        data/predictions/lstm_predictions.csv      ensemble predictions on val and test (for the dashboard)

Design choices:
  * The LSTM predicts the next-day log return, scaled by the TRAIN mean and std of that return.
    Predicted close = close * exp(predicted return).
  * Huber loss (delta = 1 in standardized units) so a few extreme days do not dominate training.
  * Early stopping watches the VALIDATION loss. The TEST split is never used for any decision.
  * Each ticker is trained with several random seeds. Results with a signal this weak vary a lot
    between seeds, so we report the seed average, its spread, and the ensemble (mean prediction).
  * Shuffling the training windows inside fit() is fine: each window is an independent sample and
    the train/val/test split itself is chronological.
  * Per-seed validation numbers are printed; per-seed TEST numbers are deliberately not, so nobody
    is tempted to pick the best seed on test.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # more reproducible numerics on CPU

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import keras

from baseline import evaluate  # same metric definitions as the baselines

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = PROJECT_ROOT / "data" / "processed"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_VERSION = "lstm_v1"
TICKERS = json.loads((PROJECT_ROOT / "config" / "universe.json").read_text(encoding="utf-8"))["tickers"]
LSTM_UNITS = 32
DENSE_UNITS = 16
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
PATIENCE = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

def build_model(window: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(LSTM_UNITS),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(DENSE_UNITS, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=keras.losses.Huber(delta=1.0))
    return model


def split_frame(d, key: str, y_key_present: bool = True) -> pd.DataFrame:
    """Small frame with the columns that baseline.evaluate expects."""
    return pd.DataFrame({
        "Close": d[f"close_{key}"],
        "target_close": d[f"target_close_{key}"],
        "target_ret": d[f"y_{key}"].astype(np.float64),
    })


# --------------------------------------------------
# 3. Train and evaluate one ticker
# --------------------------------------------------

def run_ticker(ticker: str, n_seeds: int, max_epochs: int):
    path = SEQ_DIR / f"sequences_{ticker}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/models/build_sequences.py first.")
    d = np.load(path)

    X = {s: d[f"X_{s}"] for s in ["train", "val", "test"]}
    y = {s: d[f"y_{s}"].astype(np.float64) for s in ["train", "val", "test"]}
    frames = {s: split_frame(d, s) for s in ["val", "test"]}

    # Target scaling uses TRAIN statistics only
    y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
    y_train_s = (y["train"] - y_mean) / y_std
    y_val_s = (y["val"] - y_mean) / y_std

    window, n_features = X["train"].shape[1], X["train"].shape[2]
    preds = {"val": [], "test": []}
    best_epochs, val_ratios = [], []
    zero_val_rmse = evaluate(frames["val"], np.zeros(len(frames["val"])))["rmse_ret"]

    for seed in range(n_seeds):
        t0 = time.time()
        keras.utils.set_random_seed(seed)
        model = build_model(window, n_features)
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                   restore_best_weights=True)
        hist = model.fit(X["train"], y_train_s,
                         validation_data=(X["val"], y_val_s),
                         epochs=max_epochs, batch_size=BATCH_SIZE, shuffle=True,
                         callbacks=[early_stop], verbose=0)
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

        for s in ["val", "test"]:
            preds[s].append(model.predict(X[s], verbose=0).ravel() * y_std + y_mean)

        val_rmse = evaluate(frames["val"], preds["val"][-1])["rmse_ret"]
        val_ratios.append(val_rmse / zero_val_rmse)
        best_epochs.append(best_epoch)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_DIR / f"lstm_{ticker}_s{seed}.keras")
        log.info("%s seed %d: best epoch %d, val rmse_ret_vs_zero %.4f (%.0fs)",
                 ticker, seed, best_epoch, val_ratios[-1], time.time() - t0)
        keras.backend.clear_session()

    # ---- metrics ----
    rows, pred_rows = [], []
    for s in ["val", "test"]:
        zero_rmse = evaluate(frames[s], np.zeros(len(frames[s])))["rmse_ret"]
        per_seed = pd.DataFrame([evaluate(frames[s], p) for p in preds[s]])
        per_seed["ratio"] = per_seed["rmse_ret"] / zero_rmse
        per_seed["pred_std_ratio"] = [np.std(p) / np.std(y[s]) for p in preds[s]]

        mean_row = per_seed.mean(numeric_only=True)
        rows.append({
            "ticker": ticker, "model": "lstm_seed_mean", "split": s, "n": int(mean_row["n"]),
            "mae_price": mean_row["mae_price"], "rmse_price": mean_row["rmse_price"],
            "mae_ret": mean_row["mae_ret"], "rmse_ret": mean_row["rmse_ret"],
            "dir_acc": mean_row["dir_acc"], "dir_ci95": mean_row["dir_ci95"],
            "rmse_ret_vs_zero": mean_row["ratio"], "ratio_sd": per_seed["ratio"].std(ddof=0),
            "pred_std_ratio": mean_row["pred_std_ratio"],
        })

        ens = np.mean(preds[s], axis=0)
        ens_res = evaluate(frames[s], ens)
        rows.append({
            "ticker": ticker, "model": "lstm_ensemble", "split": s, **ens_res,
            "rmse_ret_vs_zero": ens_res["rmse_ret"] / zero_rmse, "ratio_sd": np.nan,
            "pred_std_ratio": float(np.std(ens) / np.std(y[s])),
        })

        pred_rows.append(pd.DataFrame({
            "ticker": ticker, "split": s, "model_version": MODEL_VERSION,
            "forecast_date": d[f"dates_{s}"], "target_date": d[f"target_dates_{s}"],
            "close": d[f"close_{s}"], "actual_close": d[f"target_close_{s}"],
            "actual_ret": y[s], "predicted_ret": ens,
            "predicted_close": d[f"close_{s}"] * np.exp(ens),
        }))

    meta = {
        "model_version": MODEL_VERSION, "y_mean": y_mean, "y_std": y_std,
        "n_seeds": n_seeds, "best_epochs": best_epochs, "val_ratios": [float(v) for v in val_ratios],
        "train_samples": int(len(y["train"])), "window": int(window), "n_features": int(n_features),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    return rows, pd.concat(pred_rows, ignore_index=True), meta


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="random seeds per ticker")
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping usually ends sooner)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    args = parser.parse_args()

    log.info("Keras %s | seeds=%d | max epochs=%d | tickers=%s", keras.__version__, args.seeds, args.epochs, args.tickers)

    all_rows, all_preds, all_meta = [], [], {}
    for ticker in args.tickers:
        rows, preds, meta = run_ticker(ticker, args.seeds, args.epochs)
        all_rows.extend(rows)
        all_preds.append(preds)
        all_meta[ticker] = meta

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(PRED_DIR / "lstm_metrics.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(PRED_DIR / "lstm_predictions.csv", index=False)
    (MODEL_DIR / "lstm_meta.json").write_text(json.dumps(all_meta, indent=2))
    log.info("Saved metrics, predictions and meta.")

    # ---- final comparison table: baselines + LSTM ----
    parts = [metrics]
    baseline_file = PRED_DIR / "baseline_metrics.csv"
    if baseline_file.exists():
        b = pd.read_csv(baseline_file)
        parts.append(b[b["ticker"].isin(args.tickers)])
    else:
        log.warning("baseline_metrics.csv not found; showing LSTM only. Run baseline.py for the full comparison.")
    full = pd.concat(parts, ignore_index=True)

    order = {"zero_return": 0, "train_mean": 1, "ridge": 2, "lstm_seed_mean": 3, "lstm_ensemble": 4}
    show = ["ticker", "model", "mae_price", "rmse_ret", "dir_acc", "dir_ci95", "rmse_ret_vs_zero", "ratio_sd"]
    for s in ["val", "test"]:
        t = full[full["split"] == s].copy()
        t["_o"] = t["model"].map(order)
        t = t.sort_values(["ticker", "_o"])[show]
        print(f"\n===== {s.upper()} =====")
        print(t.round(4).to_string(index=False))

    print("\nHow to read this:")
    print("  rmse_ret_vs_zero < 1.0 means better than 'tomorrow = today'. ratio_sd is the spread across seeds.")
    print("  If ratio_sd is as large as the gap to 1.0, the difference is seed noise, not skill.")
    print("  pred_std_ratio (in lstm_metrics.csv) near 0 means the network collapsed to a near-constant output.")

    suspicious = metrics[(metrics["split"] == "test") & (metrics["rmse_ret_vs_zero"] < 0.95)]
    if len(suspicious):
        print("\nWARNING: test rmse_ret_vs_zero below 0.95 for:", ", ".join(suspicious["ticker"].unique()))
        print("For next-day returns this is far better than anything realistic. Check for data leakage before trusting it.")


if __name__ == "__main__":
    main()
