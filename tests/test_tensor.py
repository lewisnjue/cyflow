import numpy as np
from cyflow import Tensor
from cyflow.module import Module


def numpy_to_cyflow(np_arr):
    """Convert a NumPy ndarray to a cyflow Tensor without copying memory."""
    return Tensor(np_arr)

def cyflow_to_numpy(tensor):
    """Convert a cyflow Tensor back to a NumPy ndarray."""
    arr = np.empty(tensor.shape, dtype=np.float32)
    for idx in np.ndindex(tensor.shape):
        arr[idx] = tensor[idx]
    return arr


def test_tensor_matmul_2d_matches_numpy() -> None:
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy @ B_cy
    C_np = A_np @ B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)


def test_tensor_matmul_batched_broadcast_matches_numpy() -> None:
    A_np = np.random.randn(3, 1, 2, 4).astype(np.float32)
    B_np = np.random.randn(5, 4, 3).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy @ B_cy
    C_np = A_np @ B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_matmul_3d_matches_numpy() -> None:
    A_np = np.random.randn(2, 3, 4).astype(np.float32)
    B_np = np.random.randn(2, 4, 5).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy @ B_cy
    C_np = A_np @ B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_matmul_4d_matches_numpy() -> None:
    A_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    B_np = np.random.randn(2, 3, 5, 6).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy @ B_cy
    C_np = A_np @ B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_add_matches_numpy() -> None:
    A_np = np.random.randn(3, 4).astype(np.float32)
    B_np = np.random.randn(3, 4).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy + B_cy
    C_np = A_np + B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_sub_matches_numpy() -> None:
    A_np = np.random.randn(3, 4).astype(np.float32)
    B_np = np.random.randn(3, 4).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy - B_cy
    C_np = A_np - B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_mul_matches_numpy() -> None:
    A_np = np.random.randn(3, 4).astype(np.float32)
    B_np = np.random.randn(3, 4).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    B_cy = numpy_to_cyflow(B_np)
    C_cy = A_cy * B_cy
    C_np = A_np * B_np

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)

def test_tensor_exp() -> None:
    A_np = np.random.randn(3, 4).astype(np.float32)

    A_cy = numpy_to_cyflow(A_np)
    C_cy = A_cy.exp()
    C_np = np.exp(A_np)

    assert C_cy.shape == C_np.shape
    assert np.allclose(cyflow_to_numpy(C_cy), C_np, rtol=1e-5, atol=1e-6)


def test_tensor_slice_returns_view_for_numpy_backed_tensor() -> None:
    base_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    tensor = numpy_to_cyflow(base_np)

    view = tensor[0:2, 1:3]
    assert view.shape == (2, 2)

    view.data[0, 0] = 99.0

    assert tensor.data[0, 1] == 99.0
    assert base_np[0, 1] == 99.0


def test_tensor_slice_returns_view_for_independent_tensor() -> None:
    tensor = Tensor((2, 3), requires_grad=False)
    tensor.data[:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    view = tensor[0:2, 1:3]
    assert view.shape == (2, 2)

    view.data[1, 1] = 77.0

    assert tensor.data[1, 2] == 77.0


def test_module_apply_and_parameters_walk_nested_containers() -> None:
    class Leaf(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = Tensor((1,), requires_grad=True)

        def forward(self, *args, **kwargs):
            return None

    class Composite(Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = Leaf()
            self.right = [Leaf(), (Leaf(),)]

    composite = Composite()

    seen = []

    def mark(module):
        seen.append(module.__class__.__name__)
        return module

    composite.apply(mark)
    params = composite.parameters()

    assert len(seen) >= 3
    assert len(params) == 3