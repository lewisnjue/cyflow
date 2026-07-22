import numpy as np
import pytest

from cyflow import Tensor, Linear, SGD, mse_loss

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    optim = None


class TestLinearTrainingComparison:
    def _build_regression_problem(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 4)).astype(np.float32)
        true_w = np.array([1.2, -0.7, 0.3, 0.9], dtype=np.float32)
        y = X @ true_w + 0.05 * rng.normal(size=(120,)).astype(np.float32)
        return X, y

    def test_cyflow_linear_regression_reduces_loss(self) -> None:
        X, y = self._build_regression_problem()

        x_tensor = Tensor(X, requires_grad=False)
        y_tensor = Tensor(y.reshape(-1, 1), requires_grad=False)

        model = Linear(4, 1, bias=True)
        optimizer = SGD(model.parameters(), lr=0.01)

        for _ in range(120):
            predictions = model(x_tensor)
            loss = mse_loss(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert loss.item() < 0.3

    def test_pytorch_linear_regression_matches_cyflow_trend(self) -> None:
        if torch is None:
            pytest.skip("torch is not installed")

        X, y = self._build_regression_problem()

        x_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        torch_model = nn.Linear(4, 1, bias=True)
        torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01)

        for _ in range(120):
            predictions = torch_model(x_torch)
            loss = nn.functional.mse_loss(predictions, y_torch)
            torch_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch_optimizer.step()

        assert loss.item() < 0.3
