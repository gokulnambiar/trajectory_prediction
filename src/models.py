from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelOutputs:
    predictions: np.ndarray
    train_summary: dict[str, float]


def train_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
) -> ModelOutputs:
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    )
                ),
            ),
        ]
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test).astype(np.float32)
    return ModelOutputs(predictions=predictions, train_summary={"train_rows": float(len(x_train))})


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return self.head(outputs[:, -1, :])


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> ModelOutputs:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = LSTMRegressor(input_size=x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_features)

        final_loss = epoch_loss / len(dataset)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test).float()).cpu().numpy().astype(np.float32)

    return ModelOutputs(predictions=predictions, train_summary={"final_train_loss": float(final_loss)})
