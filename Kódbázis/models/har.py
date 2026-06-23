from arch.univariate import HARX
from arch.univariate.base import ARCHModelResult
import pandas as pd
import numpy as np

def fit_har_rv(log_returns: pd.Series) -> ARCHModelResult:
    
    rv = (log_returns.dropna() * 100) ** 2

    model = HARX(rv, lags=[1, 5, 22])

    result = model.fit(disp="off")

    return result

def forecast_har_rv_volatility(har_result: ARCHModelResult, horizon: int = 1) -> np.ndarray:

    forecast = har_result.forecast(horizon=horizon)

    variances = forecast.mean.iloc[-1].values

    volatilities = np.sqrt(
        np.maximum(variances, 0.0)
    ) / 100

    return volatilities