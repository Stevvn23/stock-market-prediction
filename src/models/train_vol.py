"""
train_vol.py

Volatility task, step 2: train one small LSTM per ticker to forecast the next 5 days of realized
volatility, and compare it with the baselines from baseline_vol.py.

Input : data/processed/sequences_<TICKER>.npz      (from build_sequences.py)
Output: models/lstmvol_<TICKER>_s<SEED>.keras      one model per ticker and seed
        models/lstm_vol_meta.json                  target scaling stats and training info (needed for inference)
        data/predictions/lstm_vol_metrics.csv      metrics on val and test, baselines included
        data/predictions/lstm_vol_predictions.csv  ensemble forecasts on val and test (for the dashboard)

Same discipline as train.py:
  * Target is standardized with TRAIN statistics only. Huber loss. Same small LSTM architecture.
  * Early stopping watches VALIDATION loss. The TEST split is never used for any decision.
  * Several seeds per ticker; we report the seed average, its spread, and the ensemble (mean prediction).
  * Per-seed validation numbers are printed; per-seed TEST numbers are deliberately not.
  * Only samples marked volok (purged at split boundaries, target known) are used.
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

from train import build_model  # same LSTM architecture and Huber loss as the return model
from baseline_vol import (ANNUALIZE, EVAL_SPLITS, TICKERS, baseline_predictions,
                          load_ticker, load_vol20_lookup, vol_metrics)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_VERSION = "lstm_vol_v1"
BATCH_SIZE = 32
PATIENCE = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Train and evaluate one ticker
# --------------------------------------------------

def run_ticker(ticker: str, lookup: pd.Series, n_seeds: int, max_epochs: int):
    data = load_ticker(ticker, lookup)
    base = baseline_predictions(data)
    base_meta = base["_meta"]

    X = {s: data[s]["X"] for s in ["train", "val", "test"]}
    y = {s: data[s]["y"] for s in ["train", "val", "test"]}

    # Target scaling uses TRAIN statistics only
    y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
    y_train_s = (y["train"] - y_mean) / y_std
    y_val_s = (y["val"] - y_mean) / y_std

    window, n_features = X["train"].shape[1], X["train"].shape[2]
    persist_rmse = {s: vol_metrics(y[s], base[s]["persist_adj"], base_meta["const"])["rmse_log"] for s in EVAL_SPLITS}

    preds = {"val": [], "test": []}
    best_epochs, val_ratios = [], []

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

        for s in EVAL_SPLITS:
            preds[s].append(model.predict(X[s], verbose=0).ravel() * y_std + y_mean)

        val_rmse = vol_metrics(y["val"], preds["val"][-1], base_meta["const"])["rmse_log"]
        val_ratios.append(val_rmse / persist_rmse["val"])
        best_epochs.append(best_epoch)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_DIR / f"lstmvol_{ticker}_s{seed}.keras")
        log.info("%s seed %d: best epoch %d, val rmse_vs_persist %.4f (%.0fs)",
                 ticker, seed, best_epoch, val_ratios[-1], time.time() - t0)
        keras.backend.clear_session()

    # ---- metrics ----
    rows, pred_rows = [], []
    for s in EVAL_SPLITS:
        # baselines
        for name, p in base[s].items():
            r = vol_metrics(y[s], p, base_meta["const"])
            rows.append({"ticker": ticker, "model": name, "split": s, **r,
                         "rmse_vs_persist": r["rmse_log"] / persist_rmse[s], "ratio_sd": np.nan})

        # LSTM seeds
        per_seed = pd.DataFrame([vol_metrics(y[s], p, base_meta["const"]) for p in preds[s]])
        per_seed["ratio"] = per_seed["rmse_log"] / persist_rmse[s]
        mean_row = per_seed.mean(numeric_only=True)
        rows.append({
            "ticker": ticker, "model": "lstm_seed_mean", "split": s, "n": int(mean_row["n"]),
            "mae_log": mean_row["mae_log"], "rmse_log": mean_row["rmse_log"],
            "r2_oos": mean_row["r2_oos"], "corr": mean_row["corr"],
            "rmse_vs_persist": mean_row["ratio"], "ratio_sd": per_seed["ratio"].std(ddof=0),
        })

        # LSTM ensemble
        ens = np.mean(preds[s], axis=0)
        er = vol_metrics(y[s], ens, base_meta["const"])
        rows.append({"ticker": ticker, "model": "lstm_ensemble", "split": s, **er,
                     "rmse_vs_persist": er["rmse_log"] / persist_rmse[s], "ratio_sd": np.nan})

        pred_rows.append(pd.DataFrame({
            "ticker": ticker, "split": s, "model_version": MODEL_VERSION,
            "forecast_date": data[s]["dates"], "horizon_days": 5,
            "actual_logvol": y[s], "predicted_logvol": ens,
            "persist_adj_logvol": base[s]["persist_adj"],
            "actual_vol_ann_pct": np.exp(y[s]) * ANNUALIZE * 100,
            "predicted_vol_ann_pct": np.exp(ens) * ANNUALIZE * 100,
            "persist_adj_vol_ann_pct": np.exp(base[s]["persist_adj"]) * ANNUALIZE * 100,
        }))

    meta = {
        "model_version": MODEL_VERSION, "y_mean": y_mean, "y_std": y_std,
        "persist_offset": base_meta["offset"], "ridge_alpha": base_meta["ridge_alpha"],
        "n_seeds": n_seeds, "best_epochs": best_epochs, "val_ratios": [float(v) for v in val_ratios],
        "train_samples": int(len(y["train"])), "window": int(window), "n_features": int(n_features),
        "horizon_days": 5, "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    return rows, pd.concat(pred_rows, ignore_index=True), meta


# --------------------------------------------------
# 3. Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="random seeds per ticker")
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping usually ends sooner)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    args = parser.parse_args()

    log.info("Keras %s | seeds=%d | max epochs=%d | tickers=%s", keras.__version__, args.seeds, args.epochs, args.tickers)

    lookup = load_vol20_lookup()
    all_rows, all_preds, all_meta = [], [], {}
    for ticker in args.tickers:
        rows, preds, meta = run_ticker(ticker, lookup, args.seeds, args.epochs)
        all_rows.extend(rows)
        all_preds.append(preds)
        all_meta[ticker] = meta

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(PRED_DIR / "lstm_vol_metrics.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(PRED_DIR / "lstm_vol_predictions.csv", index=False)
    (MODEL_DIR / "lstm_vol_meta.json").write_text(json.dumps(all_meta, indent=2))
    log.info("Saved metrics, predictions and meta.")

    order = {"const": 0, "persist_adj": 1, "ridge": 2, "lstm_seed_mean": 3, "lstm_ensemble": 4}
    show = ["ticker", "model", "mae_log", "rmse_log", "r2_oos", "corr", "rmse_vs_persist", "ratio_sd"]
    for s in EVAL_SPLITS:
        t = metrics[metrics["split"] == s].copy()
        t["_o"] = t["model"].map(order)
        t = t.sort_values(["ticker", "_o"])[show]
        print(f"\n===== {s.upper()} (log realized volatility, next 5 days) =====")
        print(t.round(4).to_string(index=False))

    print("\nHow to read this:")
    print("  rmse_vs_persist < 1.0 means better than 'volatility stays like the last 20 days'.")
    print("  ratio_sd is the spread across seeds. Neighbouring targets overlap by 4 days, so the effective")
    print("  sample is about n/5: differences of a few percent are noise.")

    suspicious = metrics[(metrics["split"] == "test") & (metrics["model"] == "lstm_ensemble") & (metrics["r2_oos"] > 0.7)]
    if len(suspicious):
        print("\nWARNING: test r2_oos above 0.7 for:", ", ".join(suspicious["ticker"].unique()))
        print("A 5-day volatility forecast this good is unusual. Check for data leakage before trusting it.")


if __name__ == "__main__":
    main()
