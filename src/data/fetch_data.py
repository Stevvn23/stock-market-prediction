
from pathlib import Path
import logging

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

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# --------------------------------------------------
# 2. Fetch historical stock data
# --------------------------------------------------

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Download historical daily OHLCV data for one ticker."""

    logging.info("Fetching %s...", ticker)

    stock = yf.Ticker(ticker)

    df = stock.history(
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Standardize the date index.
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
# 3. Validate data quality
# --------------------------------------------------

def validate_data(df: pd.DataFrame) -> None:
    """Fail if core data-quality checks do not pass."""

    if df.empty:
        raise ValueError("The combined dataset is empty.")

    if df.duplicated(subset=["Ticker", "Date"]).any():
        raise ValueError("Duplicate ticker/date records found.")

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
# 4. Check the latest completed US trading session
# --------------------------------------------------

def check_latest_session(df: pd.DataFrame) -> None:
    """
    Warn if downloaded data appears behind the latest
    completed NYSE trading session.

    This is a freshness warning, not a market-data guarantee.
    """

    now = pd.Timestamp.now(tz="America/New_York")

    calendar = mcal.get_calendar("NYSE")

    schedule = calendar.schedule(
        start_date=(now - pd.Timedelta(days=14)).date(),
        end_date=now.date(),
    )

    completed_sessions = schedule[
        schedule["market_close"] <= now
    ]

    if completed_sessions.empty:
        logging.info(
            "No completed session found in the calendar window."
        )
        return

    latest_expected = completed_sessions.index[-1].date()

    for ticker in TICKERS:
        ticker_data = df[df["Ticker"] == ticker]

        if ticker_data.empty:
            logging.warning("No rows found for %s.", ticker)
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
        else:
            logging.info(
                "%s latest available date: %s",
                ticker,
                latest_actual,
            )


# --------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------

def main() -> None:
    frames = []

    for ticker in TICKERS:
        frames.append(fetch_stock_data(ticker))

    combined = pd.concat(frames, ignore_index=True)

    validate_data(combined)
    check_latest_session(combined)

    combined.sort_values(
        ["Ticker", "Date"],
        inplace=True,
    )

    combined.to_csv(OUTPUT_FILE, index=False)

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
