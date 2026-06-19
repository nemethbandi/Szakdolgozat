from arch.univariate import HARX
from arch.univariate.base import ARCHModelResult
import pandas as pd

def fit_har_rv(log_returns: pd.Series) -> ARCHModelResult:
    
    rv = (log_returns.dropna() * 100) ** 2

    model = HARX(rv, lags=[1, 5, 22])

    result = model.fit(disp="off")

    return result

def forecast_har_rv_volatility(har_result: ARCHModelResult, horizon: int = 1) -> float:

    forecast = har_result.forecast(horizon=horizon)

    variance_forecast = float(forecast.mean.iloc[-1, horizon - 1])

    volatility_forecast = max(variance_forecast, 0.0) ** 0.5

    return volatility_forecast / 100