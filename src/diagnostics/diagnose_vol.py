"""
diagnose_vol.py

Diagnostic only. Nothing here is used to pick or tune a model.
Question: why is the "volatility stays like the last 20 days" baseline so weak on the test split?

Input : data/processed/features.csv
Output: printed tables only

Part 1  Relationship between log(trailing 20-day vol) and the outcome (log realized vol, next 5 days),
        per ticker and per split: correlation, regression slope, spread, and mean level.
Part 2  Noise floor. Realized volatility measured from only 5 daily returns is a very noisy number even
        if the true volatility were known exactly. Its log has a standard deviation of about 0.35.
Part 3  Suspicious price bars: largest daily moves, and "spike then reverse" patterns that usually
        indicate a bad data point rather than a real move.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"

# Theoretical noise: if daily returns are normal with constant volatility, 0.5*log(chi2_5 / 5) has this sd.
NOISE_SD = 0.35


def part1(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("PART 1: trailing volatility vs next-5-day realized volatility (log scale)")
    print("=" * 78)
    rows = []
    for ticker, g in df.groupby("Ticker"):
        g = g.dropna(subset=["target_logvol5"]).copy()
        g["x"] = np.log(g["vol_20"])
        for split in ["train", "val", "test", "ALL"]:
            s = g if split == "ALL" else g[g["split"] == split]
            if len(s) < 30:
                continue
            x, y = s["x"].to_numpy(), s["target_logvol5"].to_numpy()
            slope = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
            rows.append({
                "ticker": ticker, "split": split, "n": len(s),
                "corr": np.corrcoef(x, y)[0, 1],
                "slope": slope,
                "sd_x": x.std(), "sd_y": y.std(),
                "mean_y": y.mean(), "mean_y_minus_x": (y - x).mean(),
            })
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    print("\nHow to read this:")
    print("  corr        : how well the trailing volatility ranks the next 5 days. Real stock data usually shows 0.3 to 0.6.")
    print("  slope       : below 1 means volatility mean-reverts, so raw persistence overreacts to recent swings.")
    print("  sd_y        : spread of the outcome. If it is not much above the noise floor, there is little to predict.")
    print("  mean_y_minus_x : level gap between outcome and trailing volatility. A shift between train and test = regime change.")


def part2(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("PART 2: noise floor")
    print("=" * 78)
    print(f"Irreducible noise sd of a 5-day realized-vol measurement: about {NOISE_SD}")
    rows = []
    for ticker, g in df.groupby("Ticker"):
        g = g.dropna(subset=["target_logvol5"])
        for split in ["train", "val", "test"]:
            y = g.loc[g["split"] == split, "target_logvol5"]
            var_y = y.var(ddof=0)
            max_r2 = max(0.0, 1 - NOISE_SD ** 2 / var_y)
            rows.append({"ticker": ticker, "split": split, "sd_y": y.std(ddof=0),
                         "max_possible_r2": max_r2})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nmax_possible_r2 is a rough ceiling for any model on this split (even a perfect volatility forecaster).")
    print("If it is low, a weak baseline result says more about the period than about the model.")


def part3(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("PART 3: suspicious price bars")
    print("=" * 78)
    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        r = g["ret_1"]
        big = g.loc[r.abs().sort_values(ascending=False).index[:5], ["Date", "Close", "ret_1"]].copy()
        big["Date"] = big["Date"].dt.date
        big["move_pct"] = (np.exp(big["ret_1"]) - 1) * 100
        print(f"\n{ticker}: 5 largest daily moves")
        print(big[["Date", "Close", "move_pct"]].round(2).to_string(index=False))

        nxt = r.shift(-1)
        rev = g[(r.abs() > 0.06) & (nxt.abs() > 0.06) & (np.sign(r) != np.sign(nxt))]
        if len(rev):
            print(f"{ticker}: {len(rev)} day(s) with a >6% move followed by a >6% move the other way (check these):")
            print(rev["Date"].dt.date.astype(str).tolist())
        else:
            print(f"{ticker}: no spike-then-reverse patterns above 6%.")

        rng = g["hl_range"]
        print(f"{ticker}: hl_range max {rng.max():.3f}, 99th percentile {rng.quantile(0.99):.3f}")


def main() -> None:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} not found. Run build_features.py first.")
    df = pd.read_csv(FEATURES_FILE, parse_dates=["Date", "target_date"])
    if "target_logvol5" not in df.columns:
        raise ValueError("features.csv has no target_logvol5. Re-run the new build_features.py.")
    part1(df)
    part2(df)
    part3(df)


if __name__ == "__main__":
    main()
