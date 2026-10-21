"""
build_sequences.py

Phase 2, step 3: turn the feature table into LSTM-ready 3D arrays.

Input : data/processed/features.csv                (from build_features.py)
Output: data/processed/sequences_<TICKER>.npz      one file per ticker
        models/scalers.json                        per-ticker scaler statistics (needed for daily inference)

Each npz file contains, for split in {train, val, test}:
    X_<split>            float32, shape (samples, WINDOW, n_features)
    y_<split>            float32, next-day log return for each sample
    dates_<split>        date of the LAST day inside each window (the "forecast date")
    close_<split>        close on that last day (used to turn a predicted return into a price)
    target_close_<split> actual close of the next trading day
    target_dates_<split> the next trading day's date
    yv_<split>           float32, log realized volatility over the next 5 trading days (NaN if not yet known)
    volok_<split>        bool, True for samples that may be used by the volatility models
and for the single most recent day:
    X_infer, dates_infer, close_infer      (no target yet; used to forecast tomorrow)

Rules enforced here (see project brief, section 8):
  * One ticker at a time: a window never contains rows from two different tickers.
  * The scaler is fitted on that ticker's TRAIN rows only, then applied to val/test/inference.
  * A window ending on day t contains days t-WINDOW+1 ... t only. It never looks forward.
  * A sample belongs to the split of its LAST day. Early val/test windows reach back into
    earlier periods for their inputs, which is legitimate because those days are in the past.
  * Any column starting with "target_" is a label, never a feature.
  * The 5-day volatility target reaches up to 5 days into the future. The last VOL_HORIZON samples
    of train and of val would have targets that overlap the next split, so they are purged
    (volok = False). Samples whose volatility target is not known yet (NaN) are excluded as well.
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_FILE = PROJECT_ROOT / "models" / "scalers.json"

WINDOW = 60
VOL_HORIZON = 5
NON_FEATURE_COLUMNS = {"Date", "Ticker", "Close", "split"}
SPLIT_KEYS = [("train", "train"), ("val", "val"), ("test", "test"), ("inference", "infer")]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def select_feature_columns(columns) -> list:
    """Everything that is not an identifier, the split label, or a target_ column."""
    return [c for c in columns if c not in NON_FEATURE_COLUMNS and not c.startswith("target_")]


# --------------------------------------------------
# 2. Build sequences for ONE ticker
# --------------------------------------------------

def build_ticker_sequences(ticker: str, g: pd.DataFrame, feature_cols: list):
    g = g.sort_values("Date").reset_index(drop=True)
    n_rows = len(g)
    if n_rows < WINDOW + 10:
        raise ValueError(f"{ticker}: only {n_rows} rows, not enough for a window of {WINDOW}")

    F = g[feature_cols].to_numpy(dtype=np.float64)
    split = g["split"].to_numpy()

    # Scaler is fitted on TRAIN rows only.
    train_mask = split == "train"
    scaler = StandardScaler().fit(F[train_mask])
    F_scaled = scaler.transform(F).astype(np.float32)

    # Window k covers rows k ... k+WINDOW-1, so its last row is k+WINDOW-1.
    windows = sliding_window_view(F_scaled, WINDOW, axis=0).transpose(0, 2, 1)  # (n-W+1, W, f)
    end = slice(WINDOW - 1, None)

    end_split = split[end]
    end_dates = g["Date"].dt.strftime("%Y-%m-%d").to_numpy()[end]
    close = g["Close"].to_numpy(dtype=np.float64)[end]
    target_close = g["target_close"].to_numpy(dtype=np.float64)[end]
    target_ret = g["target_ret"].to_numpy(dtype=np.float64)[end]
    target_vol = g["target_logvol5"].to_numpy(dtype=np.float64)[end]
    target_dates = g["target_date"].dt.strftime("%Y-%m-%d").fillna("").to_numpy(dtype=str)[end]

    arrays = {}
    for split_name, key in SPLIT_KEYS:
        m = end_split == split_name
        arrays[f"X_{key}"] = np.ascontiguousarray(windows[m])
        arrays[f"dates_{key}"] = end_dates[m].astype(str)
        arrays[f"close_{key}"] = close[m]
        if key != "infer":
            arrays[f"y_{key}"] = target_ret[m].astype(np.float32)
            arrays[f"target_close_{key}"] = target_close[m]
            arrays[f"target_dates_{key}"] = target_dates[m]

            yv = target_vol[m]
            ok = np.isfinite(yv)
            if key in ("train", "val"):
                ok[-VOL_HORIZON:] = False  # purge: these targets overlap the next split
            arrays[f"yv_{key}"] = yv.astype(np.float32)
            arrays[f"volok_{key}"] = ok

    stats = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "fit_rows": int(train_mask.sum()),
        "fit_end_date": g.loc[train_mask, "Date"].max().strftime("%Y-%m-%d"),
    }
    return arrays, stats, F_scaled, split


# --------------------------------------------------
# 3. Validation
# --------------------------------------------------

def validate_ticker(ticker: str, arrays: dict, F_scaled: np.ndarray, split: np.ndarray,
                    g: pd.DataFrame, n_features: int) -> None:
    for key in ["train", "val", "test", "infer"]:
        X = arrays[f"X_{key}"]
        assert X.ndim == 3 and X.shape[1:] == (WINDOW, n_features), f"{ticker} {key}: bad shape {X.shape}"
        assert np.isfinite(X).all(), f"{ticker} {key}: NaN or inf in X"
        if key != "infer":
            assert np.isfinite(arrays[f"y_{key}"]).all(), f"{ticker} {key}: NaN or inf in y"
            assert len(X) == len(arrays[f"y_{key}"]), f"{ticker} {key}: X and y length differ"
            ok = arrays[f"volok_{key}"]
            yv = arrays[f"yv_{key}"]
            assert len(ok) == len(X) and len(yv) == len(X), f"{ticker} {key}: volatility arrays have wrong length"
            assert ok.sum() > 0, f"{ticker} {key}: no usable volatility samples"
            assert np.isfinite(yv[ok]).all(), f"{ticker} {key}: NaN in usable volatility targets"

    assert len(arrays["X_infer"]) == 1, f"{ticker}: expected exactly one inference window"

    # Volatility targets: only the last VOL_HORIZON-1 test samples are unknown; train/val are complete.
    assert np.isfinite(arrays["yv_train"]).all() and np.isfinite(arrays["yv_val"]).all(), \
        f"{ticker}: unexpected NaN in train/val volatility targets"
    assert int((~np.isfinite(arrays["yv_test"])).sum()) == VOL_HORIZON - 1, \
        f"{ticker}: expected exactly {VOL_HORIZON - 1} unknown test volatility targets"
    for key in ["train", "val"]:
        assert not arrays[f"volok_{key}"][-VOL_HORIZON:].any(), f"{ticker} {key}: boundary samples not purged"

    # Chronological order of samples across splits
    d = {k: arrays[f"dates_{k}"] for k in ["train", "val", "test", "infer"]}
    # ISO date strings sort chronologically, so plain string min/max is safe.
    assert max(d["train"]) < min(d["val"]), f"{ticker}: train overlaps val"
    assert max(d["val"]) < min(d["test"]), f"{ticker}: val overlaps test"
    assert max(d["test"]) < min(d["infer"]), f"{ticker}: test overlaps inference"

    # Window integrity: re-slice the scaled matrix by date and compare with the stored window.
    date_to_row = {dt: i for i, dt in enumerate(g["Date"].dt.strftime("%Y-%m-%d"))}
    for key in ["train", "val", "test", "infer"]:
        for pos in {0, len(arrays[f"X_{key}"]) - 1}:
            r = date_to_row[arrays[f"dates_{key}"][pos]]
            expected = F_scaled[r - WINDOW + 1: r + 1]
            assert np.array_equal(arrays[f"X_{key}"][pos], expected), f"{ticker} {key}: window mismatch"

    # Targets must be strictly after the window's last day
    for key in ["train", "val", "test"]:
        assert all(t > w for t, w in zip(arrays[f"target_dates_{key}"], arrays[f"dates_{key}"])), \
            f"{ticker} {key}: target not in future"


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} not found. Run src/features/build_features.py first.")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["Date", "target_date"])
    feature_cols = select_feature_columns(df.columns)
    log.info("Window = %d days, %d features: %s", WINDOW, len(feature_cols), feature_cols)

    scaler_json = {"window": WINDOW, "feature_columns": feature_cols, "scalers": {}}
    summary = []

    for ticker, g in df.groupby("Ticker", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        arrays, stats, F_scaled, split = build_ticker_sequences(ticker, g, feature_cols)
        validate_ticker(ticker, arrays, F_scaled, split, g, len(feature_cols))

        out_file = OUT_DIR / f"sequences_{ticker}.npz"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_file, **arrays)
        scaler_json["scalers"][ticker] = stats

        train_rows = F_scaled[split == "train"]
        val_rows = F_scaled[split == "val"]
        test_rows = F_scaled[split == "test"]
        summary.append({
            "ticker": ticker,
            "train": len(arrays["X_train"]),
            "val": len(arrays["X_val"]),
            "test": len(arrays["X_test"]),
            "infer": len(arrays["X_infer"]),
            "vol_train": int(arrays["volok_train"].sum()),
            "vol_val": int(arrays["volok_val"].sum()),
            "vol_test": int(arrays["volok_test"].sum()),
            "train_mean_absmax": float(np.abs(train_rows.mean(axis=0)).max()),
            "train_std_min": float(train_rows.std(axis=0).min()),
            "train_std_max": float(train_rows.std(axis=0).max()),
            "val_absmax": float(np.abs(val_rows).max()),
            "test_absmax": float(np.abs(test_rows).max()),
        })
        log.info("%s: saved %s", ticker, out_file.name)

    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCALER_FILE.write_text(json.dumps(scaler_json, indent=2))
    log.info("Saved scaler statistics to %s", SCALER_FILE)

    s = pd.DataFrame(summary)
    print("\nSamples per split (each sample = one 60-day window):")
    print(s[["ticker", "train", "val", "test", "infer"]].to_string(index=False))

    print("\nSamples usable by the volatility models (after purging boundaries / unknown targets):")
    print(s[["ticker", "vol_train", "vol_val", "vol_test"]].to_string(index=False))

    print("\nScaling sanity check:")
    print(s[["ticker", "train_mean_absmax", "train_std_min", "train_std_max",
             "val_absmax", "test_absmax"]].round(3).to_string(index=False))
    print("\nTrain should be ~0 mean and ~1 std (it was fitted there). val/test are NOT forced to that.")
    print("A large val_absmax or test_absmax (say above 8) means those periods contain values far outside")
    print("the training range, i.e. a regime shift. That is information, not a bug.")
    print("\nNext: the models load data/processed/sequences_<TICKER>.npz")


if __name__ == "__main__":
    main()
