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
    task_label, split_label, model_label, verdict   plain-English versions for people who are not technical
    generated_utc

Second output: results/backtest_series.csv  (lines for the "actual vs forecast" chart, one row per ticker, split, date)
    actual / predicted / baseline are in readable units: percent per day for returns, annualized percent for volatility.
    baseline is "no change" (0 percent) for returns and "last month's volatility" for volatility.

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
SERIES_FILE = PROJECT_ROOT / "results" / "backtest_series.csv"

TASK_LABELS = {"next_day_return": "Next-day price change", "vol_5d": "5-day price swings (volatility)"}
SPLIT_LABELS = {"val": "Validation (used while tuning)", "test": "Test (unseen data, the honest one)"}
MODEL_LABELS = {
    "zero_return": "Assume no change (tomorrow = today)",
    "train_mean": "Historical average move",
    "ridge": "Simple linear model",
    "lstm_ensemble": "LSTM (deep learning)",
    "const": "Constant (long-run average swings)",
    "persist_adj": "Last month's swings (adjusted)",
    "persist_fit": "Last month's swings (fitted)",
    "ewma_fit": "Recent-weighted swings (EWMA)",
    "har": "HAR (standard industry benchmark)",
}
BETTER_BELOW, WORSE_ABOVE = 0.97, 1.03     # inside this band the difference is treated as noise

SPLITS = ["val", "test"]
COLUMNS = ["task", "ticker", "split", "model", "model_family", "n", "rmse", "reference_model",
           "rmse_vs_reference", "dir_acc", "r2_oos", "corr", "bias",
           "task_label", "split_label", "model_label", "verdict", "generated_utc"]
SERIES_COLUMNS = ["task", "ticker", "split", "date", "actual", "predicted", "baseline"]

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


def verdict(row) -> str:
    if row["model"] == row["reference_model"]:
        return "Benchmark"
    r = row["rmse_vs_reference"]
    if pd.isna(r):
        return "n/a"
    if r < BETTER_BELOW:
        return "Better than benchmark"
    if r > WORSE_ABOVE:
        return "Worse than benchmark"
    return "About the same"


def series_table() -> pd.DataFrame:
    """Actual versus LSTM forecast over time, in readable units. Built from files the training scripts already saved."""

    frames = []

    ret_path = PRED_DIR / "lstm_predictions.csv"
    if ret_path.exists():
        r = pd.read_csv(ret_path)
        r = r[r["split"].isin(SPLITS)]
        frames.append(pd.DataFrame({
            "task": "next_day_return", "ticker": r["ticker"], "split": r["split"], "date": r["forecast_date"],
            "actual": (np.exp(r["actual_ret"]) - 1) * 100,
            "predicted": (np.exp(r["predicted_ret"]) - 1) * 100,
            "baseline": 0.0,
        }))
    else:
        log.warning("%s not found, no return series written.", ret_path)

    vol_path = PRED_DIR / "lstm_vol_predictions.csv"
    if vol_path.exists():
        v = pd.read_csv(vol_path)
        v = v[v["split"].isin(SPLITS)]
        frames.append(pd.DataFrame({
            "task": "vol_5d", "ticker": v["ticker"], "split": v["split"], "date": v["forecast_date"],
            "actual": v["actual_vol_ann_pct"], "predicted": v["predicted_vol_ann_pct"],
            "baseline": v["persist_adj_vol_ann_pct"],
        }))
    else:
        log.warning("%s not found, no volatility series written.", vol_path)

    if not frames:
        return pd.DataFrame(columns=SERIES_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[SERIES_COLUMNS].sort_values(["task", "ticker", "split", "date"]).reset_index(drop=True)


def main() -> None:
    try:
        result = pd.concat([returns_table(), volatility_table()], ignore_index=True)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    result["model_family"] = result["model"].map(family)
    result["task_label"] = result["task"].map(TASK_LABELS)
    result["split_label"] = result["split"].map(SPLIT_LABELS)
    result["model_label"] = result["model"].map(MODEL_LABELS).fillna(result["model"])
    result["verdict"] = result.apply(verdict, axis=1)
    result["generated_utc"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    result = result[COLUMNS].sort_values(["task", "split", "ticker", "model"]).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(result), OUT_FILE)

    series = series_table()
    series.to_csv(SERIES_FILE, index=False)
    log.info("Wrote %d rows to %s", len(series), SERIES_FILE)

    test = result[result["split"] == "test"]
    view = test[test["model"].str.startswith("lstm")][["task", "ticker", "model", "n", "rmse_vs_reference"]]
    print("\nLSTM on the TEST split (below 1.0 = better than the reference):")
    print(view.round(3).to_string(index=False))
    print("\nOnly the test split is an honest estimate. Do not choose or tune anything using it.")


if __name__ == "__main__":
    main()
