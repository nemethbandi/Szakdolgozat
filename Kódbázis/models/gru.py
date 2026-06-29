import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader


FEATURE_COLUMNS = [
    "log_variance",
    "return",
    "abs_return",
    "squared_return",
    "downside_squared_return",
    "rolling_log_variance_5",
    "rolling_log_variance_22",
    "rolling_log_variance_66",
]


def build_volatility_features(
    log_returns: pd.Series,
    epsilon: float = 1e-8
) -> pd.DataFrame:
    returns = log_returns.dropna().copy()

    squared_return = returns ** 2
    log_variance = np.log(squared_return + epsilon)

    features = pd.DataFrame(index=returns.index)

    features["log_variance"] = log_variance
    features["return"] = returns
    features["abs_return"] = returns.abs()
    features["squared_return"] = squared_return
    features["downside_squared_return"] = np.minimum(returns, 0.0) ** 2

    features["rolling_log_variance_5"] = np.log(
        squared_return.rolling(5).mean() + epsilon
    )

    features["rolling_log_variance_22"] = np.log(
        squared_return.rolling(22).mean() + epsilon
    )

    features["rolling_log_variance_66"] = np.log(
        squared_return.rolling(66).mean() + epsilon
    )

    features["target"] = log_variance.shift(-1)

    return features.dropna()


def create_sequences(
    log_returns: pd.Series,
    window_size: int = 30,
    epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:

    feature_df = build_volatility_features(log_returns, epsilon=epsilon)

    X_values = feature_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_values = feature_df["target"].values.astype(np.float32)

    X = []
    y = []

    for i in range(len(feature_df) - window_size + 1):
        X.append(X_values[i:i + window_size])
        y.append(y_values[i + window_size - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y


class GRUVolatilityModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_output = out[:, -1, :]
        prediction = self.fc(last_output)

        return prediction


def _standardize_train_validation(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray
):
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_train_scaled = (X_train - x_mean) / x_std
    X_val_scaled = (X_val - x_mean) / x_std

    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    return (
        X_train_scaled,
        X_val_scaled,
        y_train_scaled,
        y_val_scaled,
        x_mean,
        x_std,
        float(y_mean),
        float(y_std),
    )


def fit_gru(
    log_returns: pd.Series,
    window_size: int = 30,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    epochs: int = 150,
    learning_rate: float = 0.001,
    patience: int = 10,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    epsilon: float = 1e-8,
    device: str | None = None
) -> tuple[GRUVolatilityModel, pd.DataFrame]:

    X, y = create_sequences(
        log_returns=log_returns,
        window_size=window_size,
        epsilon=epsilon
    )

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

    (
        X_train,
        X_val,
        y_train,
        y_val,
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = _standardize_train_validation(X_train, X_val, y_train, y_val)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = GRUVolatilityModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = deepcopy(model.state_dict())
    patience_counter = 0

    loss_history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                val_predictions = model(X_val_batch)
                val_loss = criterion(val_predictions, y_val_batch)

                val_losses.append(val_loss.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))

        loss_history.append({
            "epoch": epoch + 1,
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

    model.x_mean = x_mean
    model.x_std = x_std
    model.y_mean = y_mean
    model.y_std = y_std
    model.window_size = window_size
    model.feature_columns = FEATURE_COLUMNS
    model.epsilon = epsilon
    model.device_name = device

    loss_history_df = pd.DataFrame(loss_history)

    return model, loss_history_df


def forecast_gru_volatility(
    model: GRUVolatilityModel,
    log_returns: pd.Series,
    window_size: int | None = None,
    epsilon: float | None = None
) -> float:

    if window_size is None:
        window_size = model.window_size

    if epsilon is None:
        epsilon = model.epsilon

    feature_df = build_volatility_features(log_returns, epsilon=epsilon)

    if len(feature_df) < window_size:
        raise ValueError(f"At least {window_size} feature rows are required for forecasting.")

    x = feature_df[model.feature_columns].iloc[-window_size:].values.astype(np.float32)
    x = x.reshape(1, window_size, len(model.feature_columns))

    x = (x - model.x_mean) / model.x_std

    device = getattr(model, "device_name", "cpu")
    x = torch.tensor(x, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(x)

    scaled_log_variance_forecast = prediction.item()
    log_variance_forecast = scaled_log_variance_forecast * model.y_std + model.y_mean

    variance_forecast = np.exp(log_variance_forecast)
    volatility_forecast = np.sqrt(variance_forecast)

    return float(volatility_forecast)
