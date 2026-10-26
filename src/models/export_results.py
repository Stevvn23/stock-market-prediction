"""
export_results.py

Collect the training-time evaluation tables into ONE tidy file that Power BI can read:
    results/backtest_results.csv      (committed to the main branch; it only changes when you retrain)

Inputs (written by the training and baseline scripts, local only):
    data/predictions/baseline_metrics.csv      return baselines: zero_return, train_mean, ridge
    data/predictions/lstm_metrics.csv          return LSTM: lstm_ensemble (and lstm_seed_mean)
    data/predictions/baseline_vol_metrics.csv  volatility baselines plus the saved LSTM ensemble

Output columns
    task              next_day_return | vol_5d
    ticker, split     split is val or test. TEST is the honest number; val was used for early stopping and tuning.
    model, model_family   (baseline or lstm)
    n                 samples in the split. For vol_5d the effective sample is about n / 5 (overlapping windows).
    rmse              return scale for next_day_return, log-volatility scale for vol_5d
    reference_model   zero_return (returns) or har (volatility)
    rmse_vs_reference model RMSE divided by the reference RMSE. Below 1.0 = better than the reference.
    dir_acc           direction accuracy (returns only)
    r2_oos, corr, bias   (volatility only)
    generated_utc

Run from the project root after training:   python src/models/export_results.py
"""

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
OUT_FILE = PROJECT_ROOT / "results" / "backtest_results.csv"

SPLITS = ["val", "test"]
COLUMNS = ["task", "ticker", "split", "model", "model_family", "n", "rmse", "reference_model",
           "rmse_vs_reference", "dir_acc", "r2_oos", "corr", "bias", "generated_utc"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("export_results")


def family(model: str) -> str:
    return "lstm" if model.startswith("lstm") else "baseline"


def returns_table() -> pd.DataFrame:
    frames = []
    for name in ["baseline_metrics.csv", "lstm_metrics.csv"]:
        path = PRED_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the baseline and training scripts first.")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    df = df[df["split"].isin(SPLITS) & (df["model"] != "lstm_seed_mean")].copy()
    out = pd.DataFrame({
        "task": "next_day_return",
        "ticker": df["ticker"], "split": df["split"], "model": df["model"],
        "n": df["n"], "rmse": df["rmse_ret"],
        "reference_model": "zero_return",
        "rmse_vs_reference": df["rmse_ret_vs_zero"],
        "dir_acc": df["dir_acc"],
        "r2_oos": np.nan, "corr": np.nan, "bias": np.nan,
    })
    return out


def volatility_table() -> pd.DataFrame:
    path = PRED_DIR / "baseline_vol_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run baseline_vol.py (after train_vol.py) first.")
    df = pd.read_csv(path)
    df = df[df["split"].isin(SPLITS)].copy()

    if not df["model"].str.startswith("lstm").any():
        log.warning("baseline_vol_metrics.csv has no LSTM rows. Run train_vol.py, then baseline_vol.py again.")

    out = pd.DataFrame({
        "task": "vol_5d",
        "ticker": df["ticker"], "split": df["split"], "model": df["model"],
        "n": df["n"], "rmse": df["rmse_log"],
        "reference_model": "har",
        "rmse_vs_reference": df["rmse_vs_har"],
        "dir_acc": np.nan,
        "r2_oos": df["r2_oos"], "corr": df["corr"], "bias": df["bias"],
    })
    return out


def main() -> None:
    try:
        result = pd.concat([returns_table(), volatility_table()], ignore_index=True)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    result["model_family"] = result["model"].map(family)
    result["generated_utc"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    result = result[COLUMNS].sort_values(["task", "split", "ticker", "model"]).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(result), OUT_FILE)

    test = result[result["split"] == "test"]
    view = test[test["model"].str.startswith("lstm")][["task", "ticker", "model", "n", "rmse_vs_reference"]]
    print("\nLSTM on the TEST split (below 1.0 = better than the reference):")
    print(view.round(3).to_string(index=False))
    print("\nOnly the test split is an honest estimate. Do not choose or tune anything using it.")


if __name__ == "__main__":
    main()
