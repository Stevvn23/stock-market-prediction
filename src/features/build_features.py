"""
build_features.py

Phase 2, step 1: turn raw OHLCV prices into a leakage-safe feature table.

Input : data/raw/stock_prices.csv        (Date, Ticker, Open, High, Low, Close, Volume)
Output: data/processed/features.csv      (features + target + chronological split label)

Design rules (see project brief, section 8):
  * Every feature at date t uses only data up to and including t (trailing windows only).
  * Everything is computed PER TICKER. Tickers are never mixed into one sequence.
  * Target = next trading day's log return, NOT the raw price level.
    Predicted close is recovered later as: close_t * exp(predicted_return).
    A persistence baseline ("tomorrow = today") is then simply predicted_return = 0.
  * Features are scale-free (ratios / returns), so no feature depends on the price level.
  * Split is chronological and shared by all tickers (same cutoff dates).
  * NO scaling happens here. Scalers are fitted on the training split only, later.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # the remaining 15% is the test set

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "sma5_ratio",
    "sma20_ratio",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "vol_20",
    "hl_range",
    "volume_ratio_20",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Feature engineering for ONE ticker
# --------------------------------------------------

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, scaled to 0-1. Uses only past values."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # If there were no losses at all in the window, RSI is 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi / 100.0


def build_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute features and target for a single ticker, sorted by Date."""
    g = g.sort_values("Date").reset_index(drop=True).copy()
    close = g["Close"]

    log_close = np.log(close)
    g["ret_1"] = log_close.diff(1)
    g["ret_5"] = log_close.diff(5)
    g["ret_10"] = log_close.diff(10)

    g["sma5_ratio"] = close / close.rolling(5).mean() - 1
    g["sma20_ratio"] = close / close.rolling(20).mean() - 1

    g["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    g["macd_norm"] = macd / close
    g["macd_signal_norm"] = macd_signal / close
    g["macd_hist_norm"] = (macd - macd_signal) / close

    g["vol_20"] = g["ret_1"].rolling(20).std()
    g["hl_range"] = (g["High"] - g["Low"]) / close
    g["volume_ratio_20"] = np.log(g["Volume"] / g["Volume"].rolling(20).mean())

    # Target: the NEXT trading day's information. shift(-1) is only used for labels,
    # never for features.
    g["target_date"] = g["Date"].shift(-1)
    g["target_close"] = close.shift(-1)
    g["target_ret"] = np.log(g["target_close"] / close)

    return g


# --------------------------------------------------
# 3. Chronological split (shared cutoff dates for all tickers)
# --------------------------------------------------

def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row: train / val / test / inference.

    Rows with a known target are split by date, earliest 70% train, next 15% val,
    last 15% test. The most recent row of each ticker has no target yet, so it is
    labelled "inference" (used later to produce tomorrow's forecast).

    Note: the target of the last row in a split falls on the first day of the next
    split. This is a standard, minor boundary overlap; the sequence builder keeps
    inputs strictly in the past.
    """
    labelled_dates = np.sort(df.loc[df["target_ret"].notna(), "Date"].unique())
    n = len(labelled_dates)
    train_end = labelled_dates[int(n * TRAIN_FRAC) - 1]
    val_end = labelled_dates[int(n * (TRAIN_FRAC + VAL_FRAC)) - 1]

    df["split"] = np.select(
        [
            df["target_ret"].isna(),
            df["Date"] <= train_end,
            df["Date"] <= val_end,
        ],
        ["inference", "train", "val"],
        default="test",
    )
    log.info("Split cutoffs: train <= %s, val <= %s", pd.Timestamp(train_end).date(), pd.Timestamp(val_end).date())
    return df


# --------------------------------------------------
# 4. Validation of the output
# --------------------------------------------------

def validate_output(df: pd.DataFrame) -> None:
    assert not df.duplicated(["Ticker", "Date"]).any(), "Duplicate Ticker/Date rows"

    model_rows = df[df["split"] != "inference"]
    assert model_rows[FEATURE_COLUMNS + ["target_ret"]].notna().all().all(), "NaN in model rows"

    all_features = df[FEATURE_COLUMNS]
    assert np.isfinite(all_features.to_numpy()).all(), "NaN or inf in features"

    bounds = df[df["split"].isin(["train", "val", "test"])].groupby("split")["Date"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"], "Train overlaps validation"
    assert bounds.loc["val", "max"] < bounds.loc["test", "min"], "Validation overlaps test"

    assert (df[df["split"] == "inference"].groupby("Ticker").size() == 1).all(), \
        "Each ticker should have exactly one inference row"

    log.info("Output validation passed.")


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run src/data/fetch_data.py first.")

    raw = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    log.info("Loaded %d rows for tickers: %s", len(raw), sorted(raw["Ticker"].unique()))

    parts = [build_ticker_features(g) for _, g in raw.groupby("Ticker", sort=True)]
    df = pd.concat(parts, ignore_index=True)

    # Drop the warm-up rows where trailing windows are not full yet.
    # The latest row (no target) is kept on purpose for inference.
    warmup_mask = df[FEATURE_COLUMNS].isna().any(axis=1)
    log.info("Dropping %d warm-up rows with incomplete indicators.", warmup_mask.sum())
    df = df[~warmup_mask].reset_index(drop=True)

    df = assign_split(df)
    validate_output(df)

    keep = ["Date", "Ticker", "Close"] + FEATURE_COLUMNS + ["target_date", "target_close", "target_ret", "split"]
    df = df[keep].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved %d rows to %s", len(df), OUTPUT_FILE)

    print("\nRows per ticker and split:")
    print(df.groupby(["Ticker", "split"]).size().unstack(fill_value=0))

    print("\nDate range per split:")
    print(df.groupby("split")["Date"].agg(["min", "max"]))

    print("\nTarget (next-day log return) on the TRAIN split:")
    print(df[df["split"] == "train"].groupby("Ticker")["target_ret"].describe()[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
