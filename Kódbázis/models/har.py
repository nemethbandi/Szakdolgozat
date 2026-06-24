from arch.univariate import HARX
from arch.univariate.base import ARCHModelResult
import pandas as pd


def fit_har_rv(log_returns: pd.Series) -> ARCHModelResult:
    
    rv = (log_returns.dropna() * 100) ** 2

    model = HARX(
        rv,
        lags=[1, 5, 22]
    )

    result = model.fit(disp="off")

    return result


def forecast_har_rv_volatility(har_result: ARCHModelResult, log_returns: pd.Series, horizon: int = 1) -> float:

    rv = (log_returns.dropna() * 100) ** 2

    model = HARX(
        rv,
        lags=[1, 5, 22]
    )

    fixed_result = model.fix(
        har_result.params
    )

    forecast = fixed_result.forecast(
        horizon=horizon
    )

    variance_forecast = forecast.mean.iloc[-1, 0]

    volatility_forecast = max(variance_forecast, 0.0) ** 0.5

    return volatility_forecast / 100