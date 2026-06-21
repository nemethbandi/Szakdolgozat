import pandas as pd
import torch
import torch.nn as nn
import numpy as np


def create_sequences(log_returns: pd.Series, window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:

    returns = log_returns.dropna().values

    data_length = len(returns)
    sequence_size = data_length - window_size

    X = []
    y = []

    for i in range(sequence_size):

        temp = returns[i:i+window_size]

        X.append(temp)

        y.append(
            returns[i + window_size] ** 2
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = X.reshape(-1, window_size, 1)
    y = y.reshape(-1, 1)

    return X, y


class GRUVolatilityModel(nn.Module):
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1) -> None:
        
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_output = out[:, -1, :]
        prediction = self.fc(last_output)
        
        return prediction
        

def fit_gru(log_returns: pd.Series) -> GRUVolatilityModel:

    X, y = create_sequences(log_returns)

    X_train = torch.tensor(X)
    y_train = torch.tensor(y)

    model = GRUVolatilityModel()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 100

    for epoch in range(epochs):

        predictions = model(X_train)

        loss = criterion(
            predictions,
            y_train
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def forecast_gru_volatility(model: GRUVolatilityModel, log_returns: pd.Series) -> float:

    x = np.array(
        log_returns,
        dtype=np.float32
    )

    x = x.reshape(1, 30, 1)

    x = torch.tensor(x)

    model.eval()

    with torch.no_grad():

        prediction = model(x)

    variance = prediction.item()

    volatility = variance ** 0.5

    return volatility