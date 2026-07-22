import numpy as np

from cyflow import Tensor
from cyflow.module import Module


class TestTensorCore:

    def test_tensor_matmul_2d_matches_numpy(self) -> None:
        A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy @ B_cy
        C_np = A_np @ B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_matmul_batched_broadcast_matches_numpy(self) -> None:
        A_np = np.random.randn(3, 1, 2, 4).astype(np.float32)
        B_np = np.random.randn(5, 4, 3).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy @ B_cy
        C_np = A_np @ B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_matmul_3d_matches_numpy(self) -> None:
        A_np = np.random.randn(2, 3, 4).astype(np.float32)
        B_np = np.random.randn(2, 4, 5).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy @ B_cy
        C_np = A_np @ B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)
    def test_tensor_matmul_3d_start_without_numpy(self) -> None:
        A = Tensor(shape=(2, 3, 4), requires_grad=False)
        B = Tensor(shape=(2, 4, 5), requires_grad=False)
        C = A @ B
        A_np  = A.numpy()
        B_np  = B.numpy()
        C_np = A_np @ B_np
        assert C.shape == C_np.shape
        assert np.allclose(C.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_matmul_4d_matches_numpy(self) -> None:
        A_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        B_np = np.random.randn(2, 3, 5, 6).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy @ B_cy
        C_np = A_np @ B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_add_matches_numpy(self) -> None:
        A_np = np.random.randn(3, 4).astype(np.float32)
        B_np = np.random.randn(3, 4).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy + B_cy
        C_np = A_np + B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_sub_matches_numpy(self) -> None:
        A_np = np.random.randn(3, 4).astype(np.float32)
        B_np = np.random.randn(3, 4).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy - B_cy
        C_np = A_np - B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_mul_matches_numpy(self) -> None:
        A_np = np.random.randn(3, 4).astype(np.float32)
        B_np = np.random.randn(3, 4).astype(np.float32)

        A_cy = Tensor(A_np)
        B_cy = Tensor(B_np)
        C_cy = A_cy * B_cy
        C_np = A_np * B_np

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)

    def test_tensor_exp(self) -> None:
        A_np = np.random.randn(3, 4).astype(np.float32)

        A_cy = Tensor(A_np)
        C_cy = A_cy.exp()
        C_np = np.exp(A_np)

        assert C_cy.shape == C_np.shape
        assert np.allclose(C_cy.numpy(), C_np, rtol=1e-5, atol=1e-6)


class TestTensorViewsAndModules:
    def test_tensor_slice_returns_view_for_numpy_backed_tensor(self) -> None:
        base_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        tensor = Tensor(base_np)

        view = tensor[0:2, 1:3]
        assert view.shape == (2, 2)

        view.data[0, 0] = 99.0

        assert tensor.data[0, 1] == 99.0
        assert base_np[0, 1] == 99.0

    def test_tensor_slice_returns_view_for_independent_tensor(self) -> None:
        tensor = Tensor((2, 3), requires_grad=False)
        tensor.data[:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        view = tensor[0:2, 1:3]
        assert view.shape == (2, 2)

        view.data[1, 1] = 77.0

        assert tensor.data[1, 2] == 77.0

    def test_module_apply_and_parameters_walk_nested_containers(self) -> None:
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

    def test_augmented_assignment_on_slice_updates_parent(self) -> None:
        tensor = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=False)

        tensor[0] += 2

        expected = np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        assert np.allclose(tensor.numpy(), expected)

    def test_augmented_assignment_on_scalar_updates_parent(self) -> None:
        tensor = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=False)

        tensor[0, 0] += 5

        assert tensor.data[0, 0] == 6.0
