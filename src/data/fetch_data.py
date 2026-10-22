"""
fetch_data.py

Download five years of daily OHLCV data for AAPL, MSFT and NVDA and save it as data/raw/stock_prices.csv.

Price definition: Close is split-adjusted but NOT dividend-adjusted (auto_adjust=False). Open, High and Low
follow the same convention. Keep this consistent everywhere in the project.

Built for unattended daily runs:
  * A bar from a session that has not closed yet (a partial bar, which Yahoo returns during market hours)
    is dropped, so only completed sessions are ever saved.
  * Downloads are retried with a short wait, because Yahoo sometimes refuses or rate-limits a request.
  * The output file is written only after every check has passed, and it is replaced atomically. A failed or
    stale run therefore never overwrites the last good file.
  * --strict : if any ticker is behind the latest completed NYSE session, save nothing and exit with code 3
               ("not ready yet", worth retrying later). Without --strict the same situation is only a warning.
  * --now    : act as if this UTC time were "now" (for tests), e.g. 2026-10-21T23:30:00Z

Exit codes: 0 ok, 3 not ready (stale data in strict mode), 1 any other error.
"""

from pathlib import Path
import argparse
import logging
import os
import sys
import time

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "stock_prices.csv"

TICKERS = ["AAPL", "MSFT", "NVDA"]
HISTORY_PERIOD = "5y"
MIN_ROWS_PER_TICKER = 1000          # five years is about 1255 sessions; far fewer means a truncated download
FETCH_ATTEMPTS = 3
RETRY_WAIT_SECONDS = [10, 30]       # wait before attempt 2 and attempt 3
EXIT_NOT_READY = 3

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# --------------------------------------------------
# 2. Fetch historical stock data
# --------------------------------------------------

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Download historical daily OHLCV data for one ticker (with retries)."""

    logging.info("Fetching %s...", ticker)

    df = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            break
        except Exception as exc:
            if attempt == FETCH_ATTEMPTS:
                raise
            wait = RETRY_WAIT_SECONDS[attempt - 1]
            logging.warning(
                "%s: attempt %d/%d failed (%s: %s). Retrying in %ds.",
                ticker, attempt, FETCH_ATTEMPTS, type(exc).__name__, exc, wait,
            )
            time.sleep(wait)

    # Standardize the date index (the vendor index is in New York time).
    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"

    required_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{ticker}: missing columns {missing_columns}"
        )

    # Keep only the fields needed for the first version.
    df = df[required_columns].copy()

    df = df.reset_index()
    df["Ticker"] = ticker

    df = df[
        ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    ]

    return df


# --------------------------------------------------
# 3. Latest completed session and partial bars
# --------------------------------------------------

def latest_completed_session(now: pd.Timestamp):
    """Date of the most recent NYSE session whose close is already in the past (None if not found)."""

    now_ny = now.tz_convert("America/New_York")
    calendar = mcal.get_calendar("NYSE")

    schedule = calendar.schedule(
        start_date=(now_ny - pd.Timedelta(days=14)).date(),
        end_date=now_ny.date(),
    )

    completed_sessions = schedule[schedule["market_close"] <= now]

    if completed_sessions.empty:
        return None

    return completed_sessions.index[-1].date()


def drop_partial_bars(df: pd.DataFrame, latest_expected) -> pd.DataFrame:
    """Remove rows dated after the latest completed session. Yahoo returns today's bar while the market is open."""

    if latest_expected is None:
        return df

    partial = df["Date"] > latest_expected
    if partial.any():
        logging.warning(
            "Dropping %d row(s) from a session that has not closed yet (dates: %s). "
            "Only completed sessions are saved.",
            int(partial.sum()),
            sorted({str(d) for d in df.loc[partial, "Date"]}),
        )
        df = df[~partial].copy()

    return df


# --------------------------------------------------
# 4. Validate data quality
# --------------------------------------------------

def validate_data(df: pd.DataFrame) -> None:
    """Fail if core data-quality checks do not pass."""

    if df.empty:
        raise ValueError("The combined dataset is empty.")

    if df.duplicated(subset=["Ticker", "Date"]).any():
        raise ValueError("Duplicate ticker/date records found.")

    rows = df.groupby("Ticker").size()
    for ticker in TICKERS:
        if rows.get(ticker, 0) < MIN_ROWS_PER_TICKER:
            raise ValueError(
                f"{ticker}: only {rows.get(ticker, 0)} rows, "
                f"expected at least {MIN_ROWS_PER_TICKER}. The download looks truncated."
            )

    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError("Missing values found in OHLCV data.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive stock prices found.")

    if (df["Volume"] < 0).any():
        raise ValueError("Negative trading volume found.")

    invalid_high = (
        df["High"] < df[["Open", "Low", "Close"]].max(axis=1)
    )

    invalid_low = (
        df["Low"] > df[["Open", "High", "Close"]].min(axis=1)
    )

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Inconsistent OHLC prices found.")

    # Enforce chronological ordering.
    df.sort_values(["Ticker", "Date"], inplace=True)

    logging.info("Data-quality checks passed.")


# --------------------------------------------------
# 5. Check freshness against the latest completed session
# --------------------------------------------------

def find_stale_tickers(df: pd.DataFrame, latest_expected) -> list:
    """
    Return the tickers whose latest bar is behind the latest completed NYSE session.

    This is a freshness check, not a market-data guarantee: the vendor may simply not have published the
    newest bar yet, which is why a later retry can succeed.
    """

    if latest_expected is None:
        logging.info("No completed session found in the calendar window.")
        return []

    stale = []

    for ticker in TICKERS:
        ticker_data = df[df["Ticker"] == ticker]

        if ticker_data.empty:
            logging.warning("No rows found for %s.", ticker)
            stale.append(ticker)
            continue

        latest_actual = max(ticker_data["Date"])

        if latest_actual < latest_expected:
            logging.warning(
                "%s data may be stale: latest=%s, "
                "latest completed session=%s",
                ticker,
                latest_actual,
                latest_expected,
            )
            stale.append(ticker)
        else:
            logging.info(
                "%s latest available date: %s",
                ticker,
                latest_actual,
            )

    return stale


# --------------------------------------------------
# 6. Main pipeline
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="save nothing and exit with code 3 if any ticker is behind the latest completed session")
    parser.add_argument("--now", default=None,
                        help="UTC time to act as 'now', e.g. 2026-10-21T23:30:00Z (for tests)")
    args = parser.parse_args()

    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    frames = []

    for ticker in TICKERS:
        frames.append(fetch_stock_data(ticker))

    combined = pd.concat(frames, ignore_index=True)

    latest_expected = latest_completed_session(now)
    combined = drop_partial_bars(combined, latest_expected)

    validate_data(combined)
    stale = find_stale_tickers(combined, latest_expected)

    if stale and args.strict:
        logging.error(
            "Not ready: %s behind the latest completed session (%s). Nothing was saved; "
            "the previous file is untouched. Try again later.",
            stale, latest_expected,
        )
        sys.exit(EXIT_NOT_READY)

    combined.sort_values(
        ["Ticker", "Date"],
        inplace=True,
    )

    # Write to a temporary file first, then replace, so a crash can never leave a half-written CSV.
    tmp_file = OUTPUT_FILE.with_suffix(".csv.tmp")
    combined.to_csv(tmp_file, index=False)
    os.replace(tmp_file, OUTPUT_FILE)

    logging.info("Saved %s rows to %s", len(combined), OUTPUT_FILE)

    print("\nRows per ticker:")
    print(combined.groupby("Ticker").size())

    print("\nLatest available data:")
    print(
        combined.groupby("Ticker")["Date"]
        .max()
        .to_string()
    )

    print("\nSample data:")
    print(combined.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
