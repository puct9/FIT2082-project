from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .linear_fit import Fitter


class FCNN(Fitter):
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        hidden_layer_sizes: Sequence[int],
        epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__(xs, ys)
        self.device = device
        # Store data statistics (mean, std)
        self.x_means = xs.mean(axis=0)
        self.x_stds = xs.std(axis=0)
        self.y_mean = ys.mean()
        # Fit net
        self.net = DenseNet([xs.shape[1]] + list(hidden_layer_sizes) + [1]).to(
            device
        )
        xs_t = torch.from_numpy(self.normalise_xs(xs)).float().to(device)
        ys_t = (
            torch.from_numpy(ys - self.y_mean).unsqueeze(-1).float().to(device)
        )
        self.net.fit(xs_t, ys_t, epochs=epochs)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        self.net.eval()
        # Normalise and convert to tensor
        xs_t = torch.from_numpy(self.normalise_xs(xs)).float().to(self.device)
        with torch.no_grad():
            preds: torch.Tensor = self.net(xs_t)
        return self.denormalise_ys(preds.cpu().numpy())

    def normalise_xs(self, xs: np.ndarray) -> np.ndarray:
        return (xs - self.x_means) / self.x_stds

    def denormalise_ys(self, ys: np.ndarray) -> np.ndarray:
        return ys + self.y_mean


class DenseNet(nn.Module):
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        super().__init__()
        self.fcs = nn.ModuleList(
            [
                nn.Linear(layer_size, layer_sizes[i + 1])
                for i, layer_size in enumerate(layer_sizes[:-1])
            ]
        )
        self.optim = Adam(self.parameters())

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            xs = fc(xs)
            xs = torch.relu(xs)
        # No activation on last layer
        return self.fcs[-1](xs)

    def fit(
        self, xs: torch.Tensor, ys: torch.Tensor, epochs: int = 1
    ) -> torch.Tensor:
        self.train()
        batch_size = 32
        for epoch in range(1, epochs + 1):
            order = torch.randperm(xs.shape[0])
            cum_loss = 0
            batch_iter = range(0, xs.shape[0], batch_size)
            for batch_idx in batch_iter:
                xs_batch = xs[order[batch_idx : batch_idx + batch_size]]
                ys_batch = ys[order[batch_idx : batch_idx + batch_size]]
                preds = self.forward(xs_batch)
                # Parameter update
                self.optim.zero_grad()
                loss = self.mse_loss(preds, ys_batch)
                cum_loss += loss.item()
                loss.backward()
                self.optim.step()
            print(f"Epoch {epoch}: loss={(cum_loss / len(batch_iter)):.4f}")

    def mse_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        assert preds.shape == targets.shape
        # Input shape: [ N 1 ] (x2)
        # Output shape: [ 1 ]
        return torch.square(preds - targets).mean()
