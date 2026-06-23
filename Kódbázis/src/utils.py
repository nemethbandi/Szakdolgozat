import pandas as pd
import numpy as np
from models import egarch, garch, gjr_garch, gru, har, lstm


def create_model_df() -> pd.DataFrame:

    models_df = pd.DataFrame({
        "model": [
            "GARCH",
            "EGARCH",
            "GJR-GARCH",
            "HAR",
            "LSTM",
            "GRU"
        ],
        "fit_function": [
            garch.fit_garch_1_1,
            egarch.fit_egarch_1_1,
            gjr_garch.fit_gjrgarch_1_1,
            har.fit_har_rv,
            lstm.fit_lstm,
            gru.fit_gru
        ],
        "forecast_function": [
            garch.forecast_garch_1_1_volatility,
            egarch.forecast_egarch_1_1_volatility,
            gjr_garch.forecast_gjrgarch_1_1_volatility,
            har.forecast_har_rv_volatility,
            lstm.forecast_lstm_volatility,
            gru.forecast_gru_volatility
        ]
    })

    return models_df


def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:

    returns = np.log(data["Close"]).diff()

    return pd.DataFrame({"log_returns": returns},index=data.index)