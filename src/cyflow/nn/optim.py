from typing import List

import numpy as np

from cyflow import Tensor
from cyflow.core import Module


class SGD(Module):
    """Stochastic Gradient Descent optimizer with optional momentum."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov=False,
        use_max_norm: bool = False,
        r: float = 1.0,
        grad_clip: bool = False,
        clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.use_max_norm = use_max_norm
        self.r = r
        self.grad_clip = grad_clip
        self.clip_value = clip_value
        self.velocities = [Tensor.zeros_like(p) for p in params]

    def forward(self, *args, **kwargs):
        return None

    def step(self):
        for i, param in enumerate(self.params):
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad
            if self.grad_clip:
                grad_norm = np.linalg.norm(grad.data)
                if grad_norm > self.clip_value:
                    grad.data = grad.data * (self.clip_value / grad_norm)
            if self.nesterov and self.momentum > 0:
                prev_velocity = self.velocities[i].data.copy()
                self.velocities[i].data = self.momentum * self.velocities[i].data - self.lr * grad
                param.data += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocities[i].data
            elif self.momentum > 0:
                self.velocities[i].data = self.momentum * self.velocities[i].data - self.lr * grad
                param.data += self.velocities[i].data
            else:
                param.data -= self.lr * grad

            if self.use_max_norm:
                norm = np.linalg.norm(param.data)
                if norm > self.r:
                    param.data = param.data * (self.r / norm)

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, "zero_grad"):
                param.zero_grad()

    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr
        state[f"{prefix}.momentum"] = self.momentum
        state[f"{prefix}.nesterov"] = self.nesterov
        state[f"{prefix}.use_max_norm"] = self.use_max_norm
        state[f"{prefix}.r"] = self.r
        state[f"{prefix}.grad_clip"] = self.grad_clip
        state[f"{prefix}.clip_value"] = self.clip_value
        for i, velocity in enumerate(self.velocities):
            state[f"{prefix}.velocity.{i}"] = velocity.data
        return state

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum})"

    def __str__(self) -> str:
        return f"SGD Optimizer: lr={self.lr}, momentum={self.momentum}"
