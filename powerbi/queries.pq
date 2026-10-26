// ============================================================================================================
// Power BI queries for the stock forecast project.
// How to use: Power BI Desktop > Home > Transform data > New Source > Blank Query > Advanced Editor.
// Create one blank query per block below, paste ONLY that block's code, and give the query the name in the header.
// Data comes from the public `data` branch of the GitHub repository. Web.Contents with RelativePath keeps the
// base address static, which is what allows scheduled refresh in the Power BI service later.
// ============================================================================================================


// ------------------------------------------------------------------------------------------------------------
// Query name: Prices
// Source: data/processed/features.csv  (Close plus the derived features; one row per ticker and session)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "data/data/processed/features.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"Date", type date}, {"Ticker", type text}, {"Close", type number},
        {"ret_1", type number}, {"ret_5", type number}, {"ret_10", type number},
        {"sma5_ratio", type number}, {"sma20_ratio", type number}, {"rsi_14", type number},
        {"macd_norm", type number}, {"macd_signal_norm", type number}, {"macd_hist_norm", type number},
        {"vol_20", type number}, {"hl_range", type number}, {"volume_ratio_20", type number},
        {"target_date", type date}, {"target_close", type number},
        {"target_ret", type number}, {"target_logvol5", type number}, {"split", type text}
    }, "en-US")
in
    Typed


// ------------------------------------------------------------------------------------------------------------
// Query name: Predictions
// Source: data/predictions/stock_predictions.csv  (append-only forecasts; is_live marks the real track record)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "data/data/predictions/stock_predictions.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"ticker", type text}, {"forecast_date", type date}, {"target_type", type text},
        {"horizon_days", Int64.Type}, {"target_date", type date}, {"model", type text},
        {"model_version", type text}, {"feature_version", type text}, {"train_end_date", type date},
        {"close_at_forecast", type number}, {"predicted_ret", type number}, {"predicted_close", type number},
        {"predicted_logvol", type number}, {"predicted_vol_ann_pct", type number}, {"run_utc", type datetime}
    }, "en-US"),
    IsLive = Table.TransformColumns(Typed, {{"is_live", each Text.Lower(Text.From(_)) = "true", type logical}})
in
    IsLive


// ------------------------------------------------------------------------------------------------------------
// Query name: LiveScores
// Source: data/predictions/live_scores.csv  (live forecasts that already have an outcome; empty at first)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "data/data/predictions/live_scores.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"ticker", type text}, {"forecast_date", type date}, {"target_type", type text},
        {"model", type text}, {"model_version", type text}, {"target_date", type date},
        {"predicted", type number}, {"actual", type number}, {"error", type number}
    }, "en-US")
in
    Typed


// ------------------------------------------------------------------------------------------------------------
// Query name: LiveSummary
// Source: data/predictions/live_summary.csv  (metrics per ticker, target and model; ticker "ALL" is pooled)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "data/data/predictions/live_summary.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"ticker", type text}, {"target_type", type text}, {"model", type text}, {"model_version", type text},
        {"n", Int64.Type}, {"n_days", Int64.Type}, {"n_effective", type number},
        {"first_forecast_date", type date}, {"last_forecast_date", type date},
        {"rmse", type number}, {"mae", type number}, {"bias", type number},
        {"direction_acc", type number}, {"corr", type number},
        {"reference_model", type text}, {"rmse_vs_reference", type number}, {"status", type text}
    }, "en-US")
in
    Typed


// ------------------------------------------------------------------------------------------------------------
// Query name: PipelineLogs
// Source: data/predictions/pipeline_logs.csv  (one row per forecast run)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "data/data/predictions/pipeline_logs.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"run_utc", type datetime}, {"status", type text}, {"forecast_dates", type text},
        {"rows_written", Int64.Type}, {"message", type text}
    }, "en-US")
in
    Typed


// ------------------------------------------------------------------------------------------------------------
// Query name: Dim_Ticker
// ------------------------------------------------------------------------------------------------------------
let
    Source = Table.Distinct(Table.SelectColumns(Prices, {"Ticker"})),
    Sorted = Table.Sort(Source, {{"Ticker", Order.Ascending}})
in
    Sorted


// ------------------------------------------------------------------------------------------------------------
// Query name: Dim_Date   (calendar from the first price date to two weeks after the last one)
// ------------------------------------------------------------------------------------------------------------
let
    StartDate = List.Min(Prices[Date]),
    EndDate = Date.AddDays(List.Max(Prices[Date]), 14),
    Dates = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1, 0, 0, 0)),
    AsTable = Table.FromList(Dates, Splitter.SplitByNothing(), {"Date"}, null, ExtraValues.Error),
    Typed = Table.TransformColumnTypes(AsTable, {{"Date", type date}}),
    AddYear = Table.AddColumn(Typed, "Year", each Date.Year([Date]), Int64.Type),
    AddMonth = Table.AddColumn(AddYear, "Month", each Date.ToText([Date], "yyyy-MM"), type text),
    AddWeekday = Table.AddColumn(AddMonth, "Weekday", each Date.DayOfWeekName([Date], "en-US"), type text)
in
    AddWeekday



// ------------------------------------------------------------------------------------------------------------
// Query name: Backtest
// Source: results/backtest_results.csv on the MAIN branch (training-time evaluation; changes only when you retrain)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "main/results/backtest_results.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"task", type text}, {"ticker", type text}, {"split", type text}, {"model", type text},
        {"model_family", type text}, {"n", Int64.Type}, {"rmse", type number}, {"reference_model", type text},
        {"rmse_vs_reference", type number}, {"dir_acc", type number}, {"r2_oos", type number},
        {"corr", type number}, {"bias", type number},
        {"task_label", type text}, {"split_label", type text}, {"model_label", type text}, {"verdict", type text},
        {"generated_utc", type datetime}
    }, "en-US")
in
    Typed



// ------------------------------------------------------------------------------------------------------------
// Query name: BacktestSeries
// Source: results/backtest_series.csv on the MAIN branch (actual versus LSTM forecast over time, readable units)
// ------------------------------------------------------------------------------------------------------------
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/Stevvn23/stock-market-prediction",
                     [RelativePath = "main/results/backtest_series.csv"]),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"task", type text}, {"ticker", type text}, {"split", type text}, {"date", type date},
        {"actual", type number}, {"predicted", type number}, {"baseline", type number}
    }, "en-US")
in
    Typed
