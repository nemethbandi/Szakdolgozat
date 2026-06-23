import pandas as pd
import numpy as np
from utils import create_model_df, calculate_log_returns


def train_models(data: pd.DataFrame, train_window: int = 5, forecast_window: int = 1) -> pd.DataFrame:

    start_year = data.index[0].year

    log_data = calculate_log_returns(data)

    model_df = create_model_df()

    results = []

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

        for _, model_row in model_df.iterrows():

            model_name = model_row["model"]
            fit_function = model_row["fit_function"]
            forecast_function = model_row["forecast_function"]

            fitted_model = fit_function(train_data["log_returns"])

            if model_name in ["GARCH", "EGARCH", "GJR-GARCH", "HAR"]:

                volatility_forecasts = forecast_function(
                    fitted_model,
                    horizon=len(forecast_data)
                )

                for forecast_date, volatility_forecast in zip(
                    forecast_data.index,
                    volatility_forecasts
                ):

                    results.append({
                        "date": forecast_date,
                        "model": model_name,
                        "forecast_volatility": volatility_forecast
                    })

            else:

                full_period_data = pd.concat(
                    [
                        train_data,
                        forecast_data
                    ]
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

    return wide_results_df