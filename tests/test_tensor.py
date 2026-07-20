import numpy as np
from cyflow import Tensor


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