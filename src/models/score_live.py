"""
score_live.py

Score the LIVE forecasts against what actually happened.

Input : data/predictions/stock_predictions.csv   (only rows with is_live == True are ever scored)
        data/processed/features.csv              (provides the realized outcomes, see below)
Output: data/predictions/live_scores.csv    one row per scored forecast (a derived file, rebuilt on every run)
        data/predictions/live_summary.csv   metrics per ticker / target / model, plus a pooled "ALL" row

Why features.csv supplies the outcomes
  features.csv is rebuilt from one consistent price download on every run, and its target columns are defined in
  one place (build_features.py): target_ret is the next-day log return and target_logvol5 is the log realized
  volatility of the next 5 daily returns. Scoring against the same definitions the models were trained on, from the
  same download, avoids mixing price levels from different downloads (a stock split rescales history).

A forecast is scorable once its outcome exists (next session for close_1d, fifth session for vol_5d).
Rows with is_live == False (development backfills) are excluded, always. There is no option to include them.

Metrics
  close_1d (log return) : rmse, mae, bias, direction accuracy (skipped for zero forecasts), rmse relative to the
                          persistence forecast (zero return). Below 1.0 means better than "tomorrow equals today".
  vol_5d  (log vol)     : rmse, mae, bias, corr, rmse relative to the HAR forecast. Below 1.0 means better than HAR.
  The 5-day windows overlap, so 5 consecutive vol_5d forecasts carry roughly one independent observation:
  n_effective = n / 5 for vol_5d.
  status = "accumulating" until the number of distinct forecast DAYS (n_days) reaches MIN_DAYS_FOR_READING, then
  "readable". Days are counted, not rows: the pooled "ALL" row holds one forecast per ticker per day, and tickers
  on the same day move together, so three tickers do not make three independent days.
  "readable" only means the sample is large enough to look at; it does not mean a difference between models is
  statistically significant.
"""

from pathlib import Path
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_FILE = PROJECT_ROOT / "data" / "predictions" / "stock_predictions.csv"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
SCORES_FILE = PROJECT_ROOT / "data" / "predictions" / "live_scores.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "predictions" / "live_summary.csv"

VOL_HORIZON = 5
REFERENCE = {"close_1d": "persistence", "vol_5d": "har"}      # models the others are compared against
MIN_DAYS_FOR_READING = {"close_1d": 60, "vol_5d": 120}
MIN_N_FOR_CORR = 20

SCORE_COLUMNS = ["ticker", "forecast_date", "target_type", "model", "model_version", "target_date",
                 "predicted", "actual", "error"]
SUMMARY_COLUMNS = ["ticker", "target_type", "model", "model_version", "n", "n_days", "n_effective", "first_forecast_date",
                   "last_forecast_date", "rmse", "mae", "bias", "direction_acc", "corr", "reference_model",
                   "rmse_vs_reference", "status"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("score_live")


# --------------------------------------------------
# 1. Load and join
# --------------------------------------------------

def load_live_predictions(path: Path) -> pd.DataFrame:
    """Read the forecast file and keep ONLY live rows."""

    pred = pd.read_csv(path, dtype={"forecast_date": str, "target_date": str, "train_end_date": str})

    is_live = pred["is_live"].astype(str).str.strip().str.lower().eq("true")
    dropped = int((~is_live).sum())
    if dropped:
        log.info("Ignoring %d row(s) with is_live != True.", dropped)

    return pred[is_live].copy()


def realized_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (ticker, forecast_date) with the realized outcomes and the date each horizon ends on.

    close_1d ends on the next session; vol_5d ends on the fifth session after the forecast date. Both end
    dates are derived from the session list in features.csv so they can be checked against the stored target_date.
    """

    rows = []
    for ticker, g in features.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        dates = g["Date"].dt.strftime("%Y-%m-%d")

        out = pd.DataFrame({
            "ticker": ticker,
            "forecast_date": dates,
            "actual_close_1d": g["target_ret"],
            "actual_vol_5d": g["target_logvol5"],
            "end_close_1d": dates.shift(-1),
            "end_vol_5d": dates.shift(-VOL_HORIZON),
        })
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def build_scores(pred: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join live forecasts to outcomes. Keeps only forecasts whose outcome exists and whose dates agree."""

    merged = pred.merge(outcomes, on=["ticker", "forecast_date"], how="left")

    merged["actual"] = np.where(merged["target_type"] == "close_1d", merged["actual_close_1d"], merged["actual_vol_5d"])
    merged["end_expected"] = np.where(merged["target_type"] == "close_1d", merged["end_close_1d"], merged["end_vol_5d"])
    merged["predicted"] = np.where(merged["target_type"] == "close_1d", merged["predicted_ret"], merged["predicted_logvol"])

    pending = merged["actual"].isna() | merged["end_expected"].isna()

    # The stored target_date must equal the session the features file says the horizon ends on.
    mismatch = ~pending & (merged["end_expected"].astype(str) != merged["target_date"].astype(str))
    if mismatch.any():
        log.warning("%d forecast(s) have a target_date that disagrees with the session calendar in features.csv "
                    "and are NOT scored: %s", int(mismatch.sum()),
                    merged.loc[mismatch, ["ticker", "forecast_date", "target_type"]].head(5).to_dict("records"))

    scored = merged[~pending & ~mismatch].copy()
    scored["error"] = scored["predicted"] - scored["actual"]

    log.info("Live forecasts: %d total, %d scored, %d pending, %d rejected.",
             len(merged), len(scored), int(pending.sum()), int(mismatch.sum()))

    return scored[SCORE_COLUMNS].sort_values(["ticker", "target_type", "model", "forecast_date"]).reset_index(drop=True)


# --------------------------------------------------
# 2. Summary
# --------------------------------------------------

def _rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Metrics per (ticker, target_type, model), plus pooled rows with ticker == 'ALL'."""

    if scores.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    pooled = scores.copy()
    pooled["ticker"] = "ALL"
    both = pd.concat([scores, pooled], ignore_index=True)

    rows = []
    for (ticker, target_type, model), g in both.groupby(["ticker", "target_type", "model"]):
        n = len(g)
        n_days = int(g["forecast_date"].nunique())
        row = {
            "ticker": ticker, "target_type": target_type, "model": model,
            "model_version": "/".join(sorted(g["model_version"].astype(str).unique())),
            "n": n,
            "n_days": n_days,
            "n_effective": n_days if target_type == "close_1d" else round(n_days / VOL_HORIZON, 1),
            "first_forecast_date": g["forecast_date"].min(),
            "last_forecast_date": g["forecast_date"].max(),
            "rmse": _rmse(g["error"]),
            "mae": float(g["error"].abs().mean()),
            "bias": float(g["error"].mean()),
            "direction_acc": np.nan,
            "corr": np.nan,
            "reference_model": REFERENCE[target_type],
            "rmse_vs_reference": np.nan,
            "status": "readable" if n_days >= MIN_DAYS_FOR_READING[target_type] else "accumulating",
        }

        if target_type == "close_1d":
            nonzero = g[g["predicted"] != 0]
            if len(nonzero):
                row["direction_acc"] = float((np.sign(nonzero["predicted"]) == np.sign(nonzero["actual"])).mean())
        elif n >= MIN_N_FOR_CORR and g["predicted"].std() > 0 and g["actual"].std() > 0:
            row["corr"] = float(np.corrcoef(g["predicted"], g["actual"])[0, 1])

        # Compare with the reference model on exactly the same forecasts.
        if model != REFERENCE[target_type]:
            common = _paired_with_reference(scores, target_type, model, None if ticker == "ALL" else ticker)
            if len(common):
                ref_rmse = _rmse(common["error_ref"])
                if ref_rmse > 0:
                    row["rmse_vs_reference"] = _rmse(common["error"]) / ref_rmse

        rows.append(row)

    out = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    order = {"close_1d": 0, "vol_5d": 1}
    out["_o"] = out["target_type"].map(order)
    out["_t"] = out["ticker"].map(lambda t: (1, t) if t == "ALL" else (0, t))
    out = out.sort_values(["_o", "_t", "model"]).drop(columns=["_o", "_t"]).reset_index(drop=True)
    return out


def _paired_with_reference(scores: pd.DataFrame, target_type: str, model: str, ticker=None) -> pd.DataFrame:
    """A model's errors paired with the reference model's errors on the same ticker and forecast date."""

    base = scores[scores["target_type"] == target_type]
    if ticker is not None:
        base = base[base["ticker"] == ticker]

    a = base[base["model"] == model]
    b = base[base["model"] == REFERENCE[target_type]]
    return a.merge(b[["ticker", "forecast_date", "error"]], on=["ticker", "forecast_date"], suffixes=("", "_ref"))


# --------------------------------------------------
# 3. Main
# --------------------------------------------------

def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()

    for f in (PRED_FILE, FEATURES_FILE):
        if not f.exists():
            log.error("%s not found.", f)
            sys.exit(1)

    pred = load_live_predictions(PRED_FILE)
    features = pd.read_csv(FEATURES_FILE, parse_dates=["Date"])

    scores = build_scores(pred, realized_outcomes(features))
    summary = summarize(scores)

    write_atomic(scores, SCORES_FILE)
    write_atomic(summary, SUMMARY_FILE)
    log.info("Wrote %d score row(s) and %d summary row(s).", len(scores), len(summary))

    if summary.empty:
        print("\nNothing to score yet: no live forecast has reached its outcome date.")
        return

    show = summary[summary["ticker"] == "ALL"]
    print("\nPooled live track record (all tickers, live forecasts only):")
    print(show[["target_type", "model", "n", "n_days", "n_effective", "rmse", "rmse_vs_reference", "direction_acc",
                "status"]].round(4).to_string(index=False))
    print("\nstatus 'accumulating' means too few days to read anything into the numbers.")


if __name__ == "__main__":
    main()
