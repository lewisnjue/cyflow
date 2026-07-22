from cyflow import Tensor
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy.typing as npt
from cyflow.core import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32), requires_grad=True)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "x must be a `Tensor`"
        assert x.shape[-1] == self.in_features, f"Input size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        if self.has_bias:
            return x @ self.weight + self.bias
        return x @ self.weight

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

    def __str__(self):
        return self.__repr__()
