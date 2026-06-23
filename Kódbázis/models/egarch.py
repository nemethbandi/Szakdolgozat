from arch import arch_model
from arch.univariate.base import ARCHModelResult
import pandas as pd
import numpy as np

def fit_egarch_1_1(log_returns: pd.Series) -> ARCHModelResult:

    returns = log_returns.dropna() * 100

    model = arch_model(
        returns,
        mean="Constant",
        vol="EGARCH",
        p=1,
        q=1,
        dist="normal"
    )

    result = model.fit(disp="off")

    return result

def forecast_egarch_1_1_volatility(garch_result: ARCHModelResult, horizon: int = 1) -> float:
    
    forecast = garch_result.forecast(
        horizon=horizon,
        method="simulation",
        simulations=1000
    )

    variances = forecast.variance.iloc[-1].values

    volatilities = np.sqrt(variances) / 100

    return volatilities