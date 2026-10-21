"""
build_features.py

Phase 2, step 1: turn raw OHLCV prices into a leakage-safe feature table.

Input : data/raw/stock_prices.csv        (Date, Ticker, Open, High, Low, Close, Volume)
Output: data/processed/features.csv      (features + targets + chronological split label)

Design rules (see project brief, section 8):
  * Every feature at date t uses only data up to and including t (trailing windows only).
  * Everything is computed PER TICKER. Tickers are never mixed into one sequence.
  * Return target  = next trading day's log return, NOT the raw price level.
    Predicted close is recovered later as: close_t * exp(predicted_return).
    A persistence baseline ("tomorrow = today") is then simply predicted_return = 0.
  * Volatility target = log of the realized volatility over the NEXT 5 trading days,
    0.5 * log(mean(r_{t+1} ... r_{t+5} squared)). It only uses future returns, never features.
  * Features are scale-free (ratios / returns), so no feature depends on the price level.
  * Split is chronological and shared by all tickers (same cutoff dates).
  * NO scaling happens here. Scalers are fitted on the training split only, later.
  * Every column that is a label or comes from the future starts with "target_".
    Downstream scripts treat any column starting with "target_" as NOT a feature.
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
VOL_HORIZON = 5  # trading days used for the forward realized-volatility target

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
    """Compute features and targets for a single ticker, sorted by Date."""
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

    # Targets: the FUTURE. shift(-1) etc. are only used for labels, never for features.
    g["target_date"] = g["Date"].shift(-1)
    g["target_close"] = close.shift(-1)
    g["target_ret"] = np.log(g["target_close"] / close)

    # Forward realized volatility over the next VOL_HORIZON days (log scale).
    # ret_1 at t+k is the return from t+k-1 to t+k, so k = 1..5 covers the next 5 sessions.
    fwd_sq = sum(g["ret_1"].shift(-k) ** 2 for k in range(1, VOL_HORIZON + 1)) / VOL_HORIZON
    g["target_logvol5"] = 0.5 * np.log(fwd_sq + 1e-12)

    return g


# --------------------------------------------------
# 3. Chronological split (shared cutoff dates for all tickers)
# --------------------------------------------------

def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row: train / val / test / inference.

    Rows with a known next-day return are split by date, earliest 70% train, next 15% val,
    last 15% test. The most recent row of each ticker has no target yet, so it is
    labelled "inference" (used later to produce tomorrow's forecast).

    Note: the target of the last row in a split falls on the first day of the next
    split. For the 1-day return this is a standard, minor boundary overlap. The 5-day
    volatility target overlaps by up to 4 days, so build_sequences.py purges those
    boundary samples for the volatility task.
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

    # The volatility target needs VOL_HORIZON future returns, so exactly the last VOL_HORIZON rows lack it.
    missing_vol = df.groupby("Ticker")["target_logvol5"].apply(lambda s: int(s.isna().sum()))
    assert (missing_vol == VOL_HORIZON).all(), f"Unexpected missing volatility targets: {missing_vol.to_dict()}"
    assert np.isfinite(df["target_logvol5"].dropna().to_numpy()).all(), "inf in volatility target"

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
    # The latest rows (no target yet) are kept on purpose for inference.
    warmup_mask = df[FEATURE_COLUMNS].isna().any(axis=1)
    log.info("Dropping %d warm-up rows with incomplete indicators.", warmup_mask.sum())
    df = df[~warmup_mask].reset_index(drop=True)

    df = assign_split(df)
    validate_output(df)

    keep = (["Date", "Ticker", "Close"] + FEATURE_COLUMNS +
            ["target_date", "target_close", "target_ret", "target_logvol5", "split"])
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

    train = df[df["split"] == "train"]
    ann = train.groupby("Ticker")["target_logvol5"].apply(lambda s: float(np.exp(s.mean()) * np.sqrt(252) * 100))
    print("\nTypical forward 5-day realized volatility on the TRAIN split (annualized %):")
    print(ann.round(1).to_string())


if __name__ == "__main__":
    main()
