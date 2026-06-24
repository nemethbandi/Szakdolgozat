from arch import arch_model
from arch.univariate.base import ARCHModelResult
import pandas as pd
import numpy as np


def fit_egarch_1_1(log_returns: pd.Series) -> ARCHModelResult:

    returns = log_returns.dropna() * 100

    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    result = model.fit(disp="off")

    return result


def forecast_egarch_1_1_volatility(garch_result: ARCHModelResult, log_returns: pd.Series, horizon: int = 1) -> float:

    returns = log_returns.dropna() * 100

    model = arch_model(
        returns,
        mean="Constant",
        vol="EGARCH",
        p=1,
        q=1,
        dist="normal"
    )

    fixed_result = model.fix(
        garch_result.params
    )

    forecast = fixed_result.forecast(
        horizon=horizon
    )

    variance_forecast = forecast.variance.iloc[-1, 0]
    volatility_forecast = variance_forecast ** 0.5

    return volatility_forecast / 100