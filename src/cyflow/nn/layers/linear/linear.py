import cyflow
import numpy as np
class Linear:
    def __init__(self, input_dim, output_dim, device="cpu", bias: bool = True):
        assert device in ["cpu", "cuda"] or isinstance(device, int), "Device must be either 'cpu' or 'cuda' or an integer constant"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Map string device names to cyflow constants when creating tensors
        dev = device
        if isinstance(device, str):
            dev = cyflow.CPU if device == "cpu" else cyflow.CUDA

        self.weights = cyflow.tensor(data=np.random.randn(input_dim, output_dim).tolist(), shape=(input_dim, output_dim), device=dev, requires_grad=True)
        if bias:
            self.bias = cyflow.tensor(data=np.zeros(output_dim).tolist(), shape=(output_dim,), device=dev, requires_grad=True)
        else:
            self.bias = None
    
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        assert x.shape[-1] == self.input_dim, f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
        output = x @ self.weights
        if self.bias is not None:
            output = output + self.bias
        return output
