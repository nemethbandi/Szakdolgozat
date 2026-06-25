import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader


def create_log_volatility_series(log_returns: pd.Series, epsilon: float = 1e-8) -> pd.Series:

    volatility = log_returns.abs()
    log_volatility = np.log(volatility + epsilon)

    return log_volatility.dropna()


def create_sequences(log_returns: pd.Series, window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:

    log_volatility = create_log_volatility_series(log_returns)

    values = log_volatility.values

    data_length = len(values)
    sequence_size = data_length - window_size

    X = []
    y = []

    for i in range(sequence_size):

        temp = values[i:i + window_size]

        X.append(temp)

        y.append(
            values[i + window_size]
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = X.reshape(-1, window_size, 1)
    y = y.reshape(-1, 1)

    return X, y


class LSTMVolatilityModel(nn.Module):

    def __init__(self, input_size: int = 1, hidden_size: int = 16, num_layers: int = 1, output_size: int = 1) -> None:

        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        prediction = self.fc(last_output)

        return prediction


def fit_lstm(log_returns: pd.Series, window_size: int = 30, train_ratio: float = 0.8, batch_size: int = 32, epochs: int = 100, learning_rate: float = 0.001) -> tuple[LSTMVolatilityModel, pd.DataFrame]:

    X, y = create_sequences(log_returns, window_size=window_size)

    if len(X) < 2:

        raise ValueError("Not enough observations to create training and validation sequences.")

    if not 0 < train_ratio < 1:

        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(X) * train_ratio)

    if split_index == 0 or split_index == len(X):

        raise ValueError("train_ratio must produce non-empty training and validation sets.")

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMVolatilityModel()

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = deepcopy(model.state_dict())
    patience = 5
    patience_counter = 0

    loss_history = []

    for epoch in range(epochs):

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_val_batch, y_val_batch in val_loader:

                val_predictions = model(X_val_batch)

                val_loss = criterion(val_predictions, y_val_batch)

                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        loss_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "validation_loss": avg_val_loss
        })

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            break

    model.load_state_dict(best_model_state)

    loss_history_df = pd.DataFrame(loss_history)

    return model, loss_history_df


def forecast_lstm_volatility(model: LSTMVolatilityModel, log_returns: pd.Series, window_size: int = 30, epsilon: float = 1e-8) -> float:

    log_volatility = create_log_volatility_series(log_returns, epsilon=epsilon)

    if len(log_volatility) < window_size:

        raise ValueError(f"At least {window_size} observations are required for forecasting.")

    x = np.array(log_volatility.iloc[-window_size:], dtype=np.float32)

    x = x.reshape(1, window_size, 1)

    x = torch.tensor(x, dtype=torch.float32)

    model.eval()

    with torch.no_grad():

        prediction = model(x)

    log_volatility_forecast = prediction.item()

    volatility_forecast = np.exp(log_volatility_forecast)

    return float(volatility_forecast)
