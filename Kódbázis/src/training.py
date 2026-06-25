import pandas as pd
from utils import create_model_df, calculate_log_returns


def train_models(data: pd.DataFrame, train_window: int = 5, forecast_window: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:

    start_year = data.index[0].year

    log_data = calculate_log_returns(data)

    model_df = create_model_df()

    results = []
    loss_results = []

    for i in range((data.index[-1].year - data.index[0].year) - train_window + 1):

        train_start = start_year
        train_end = start_year + train_window

        forecast_start = train_end
        forecast_end = forecast_start + forecast_window

        train_data = log_data[
            (log_data.index.year >= train_start) &
            (log_data.index.year < train_end)
        ]

        forecast_data = log_data[
            (log_data.index.year >= forecast_start) &
            (log_data.index.year < forecast_end)
        ]

        print(f"Train: {train_start}-{train_end-1}")
        print(f"Forecast: {forecast_start}")

        full_period_data = pd.concat(
            [
                train_data,
                forecast_data
            ]
        )

        for _, model_row in model_df.iterrows():

            model_name = model_row["model"]
            fit_function = model_row["fit_function"]
            forecast_function = model_row["forecast_function"]

            if model_name in ["LSTM", "GRU"]:

                fitted_model, model_loss_df = fit_function(
                    train_data["log_returns"]
                )

                model_loss_df["model"] = model_name
                model_loss_df["train_start"] = train_start
                model_loss_df["train_end"] = train_end - 1
                model_loss_df["forecast_year"] = forecast_start

                loss_results.append(model_loss_df)

            else:

                fitted_model = fit_function(
                    train_data["log_returns"]
                )

            for forecast_date in forecast_data.index:

                available_data = full_period_data[
                    full_period_data.index < forecast_date
                ]["log_returns"]

                volatility_forecast = forecast_function(
                    fitted_model,
                    available_data
                )

                results.append({
                    "date": forecast_date,
                    "model": model_name,
                    "forecast_volatility": volatility_forecast
                })

        start_year += 1

    results_df = pd.DataFrame(results)

    wide_results_df = results_df.pivot(
        index="date",
        columns="model",
        values="forecast_volatility"
    )

    if len(loss_results) > 0:

        loss_df = pd.concat(
            loss_results,
            ignore_index=True
        )

    else:

        loss_df = pd.DataFrame()

    return wide_results_df, loss_df