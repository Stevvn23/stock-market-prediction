# Power BI setup

Power BI Desktop (free, Windows). Everything reads from the public `data` branch, so no login is needed.

## 1. Load the data
1. Home > Transform data > New Source > Blank Query. Open Advanced Editor.
2. Paste ONE block from `queries.pq` (the code under each "Query name" header, not the comment lines), click Done,
   rename the query to the name in the header. Repeat for all 7.
3. If Power BI asks about data privacy or credentials: choose Anonymous, and set the privacy level to Public.
4. Close and Apply.

## 2. Relationships (Model view)
| From (one) | To (many) |
|---|---|
| Dim_Ticker[Ticker] | Prices[Ticker] |
| Dim_Ticker[Ticker] | Predictions[ticker] |
| Dim_Ticker[Ticker] | LiveScores[ticker] |
| Dim_Date[Date] | Prices[Date] |
| Dim_Date[Date] | Predictions[forecast_date] |
| Dim_Date[Date] | LiveScores[forecast_date] |

Do NOT relate LiveSummary (it contains a pooled "ALL" ticker) or PipelineLogs. Use their own columns as slicers.
Set every relationship to single direction, one to many.

## 3. Measures
Paste each measure from `measures.dax` (Home > New measure). A measure is the text from its name to the next blank line.

## 4. Pages
1. **Market Overview**: slicer Dim_Ticker[Ticker]. Cards: Latest Close, 1D Return, 5D Return, Volatility 20D
   Annualized, RSI 14. Line chart: Prices[Date] and Close. Line chart of vol_20 by date. Use a "Date" slicer from Dim_Date.
2. **Forecasts**: table with Dim_Ticker[Ticker], Latest Live Forecast Date, Latest Target Date, Forecast Close (LSTM),
   Forecast Move % (LSTM), Forecast Vol % (LSTM), Forecast Vol % (HAR). Add a text note: "Forecasts are experimental,
   not advice. Next-day return forecasts showed no edge in testing."
3. **Live Track Record**: slicers LiveScores[target_type] and ticker. Table by model: Scored Forecasts, RMSE,
   RMSE vs Reference, Direction Accuracy (returns only), Bias. Card: Track Record Status. A line chart of cumulative
   squared error by forecast_date per model comes later, once there are 20+ days.
4. **Backtest vs Live** (optional, later): validation and test numbers from the README next to the live numbers.
5. **Pipeline Monitoring**: cards Pipeline Health, Last Healthy Run (UTC), Total Runs, Failed Runs. Table of PipelineLogs
   sorted by run_utc descending.

## 5. Refresh
Home > Refresh pulls the newest CSVs. GitHub caches raw files for about 5 minutes. A scheduled cloud refresh needs a
Power BI account and workspace that allows it, and is optional: screenshots plus the .pbix in the repo are enough
for the portfolio.

## Things to know
- LiveScores and LiveSummary are empty until forecasts reach their outcome date. Empty visuals are expected at first.
- Never put backfilled rows (is_live = FALSE) in the track record. The measures above only use live rows.
- The vol_5d scores are on the log-volatility scale. Do not average them with return errors.
