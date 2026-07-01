import pandas as pd
import numpy as np

def compute_strategy(stock_data: pd.DataFrame, volatility_dataframe: pd.DataFrame, target_volatility: float = 0.15, max_leverage: float = 1.0, initial_amount: float = 1_000_000) -> pd.DataFrame:

    df = stock_data.copy()

    df["return"] = df["Close"].pct_change()

    df["forecast_volatility_daily"] = volatility_dataframe.squeeze()

    df["forecast_volatility"] = df["forecast_volatility_daily"] * np.sqrt(252)

    df["weight"] = target_volatility / df["forecast_volatility"]

    df["weight"] = df["weight"].clip(lower=0.0, upper=max_leverage)

    df["strategy_return"] = df["weight"] * df["return"]

    df["portfolio_value"] = initial_amount * (1 + df["strategy_return"]).cumprod()

    df["buy_hold_value"] = initial_amount * (1 + df["return"]).cumprod()

    df = df.dropna()

    return df
    
    