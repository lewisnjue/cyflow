import numpy as np
import pytest
import cyflow


def approx_list(tensor):
    return np.array(tensor._to_nested_list())


class TestMatmulForward:
    def test_matmul_2d(self):
        A = cyflow.tensor([[1, 2, 3], [4, 5, 6]], device=cyflow.CPU)
        B = cyflow.tensor([[7, 8], [9, 10], [11, 12]], device=cyflow.CPU)

        C = A @ B
        expected = np.matmul(np.array(A._to_nested_list()), np.array(B._to_nested_list()))

        assert C.shape == (2, 2)
        assert np.allclose(approx_list(C), expected)

    def test_matmul_batched(self):
        A = cyflow.tensor(np.random.randn(4, 2, 3).tolist(), device=cyflow.CPU)
        B = cyflow.tensor(np.random.randn(1, 3, 5).tolist(), device=cyflow.CPU)

        C = A @ B
        expected = np.matmul(np.array(A._to_nested_list()), np.array(B._to_nested_list()))

        assert C.shape == (4, 2, 5)
        assert np.allclose(approx_list(C), expected, atol=1e-6)


class TestMatmulBackward:
    def test_matmul_backward_simple(self):
        # A: (2,3), B: (3,2) -> C: (2,2)
        A = cyflow.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=cyflow.CPU, requires_grad=True)
        B = cyflow.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device=cyflow.CPU, requires_grad=True)

        C = A @ B
        # simple scalar loss: sum(C)
        loss = C
        # manually create gradient of same shape filled with 1.0
        grad_out = cyflow.tensor([[1.0, 1.0], [1.0, 1.0]], device=cyflow.CPU)

        # propagate
        C.grad_fn.next_functions = [A.grad_fn, B.grad_fn] if C.grad_fn else []
        # call backward via MatmulBackward.apply semantics
        mb = cyflow.autograd.MatmulBackward(A, B)
        gA, gB = mb.apply(grad_out)

        # expected gradients: dA = grad_out @ B^T, dB = A^T @ grad_out
        expected_gA = np.matmul(np.array(grad_out._to_nested_list()), np.array(B._to_nested_list()).T)
        expected_gB = np.matmul(np.array(A._to_nested_list()).T, np.array(grad_out._to_nested_list()))

        assert gA.shape == A.shape
        assert gB.shape == B.shape
        assert np.allclose(np.array(gA._to_nested_list()), expected_gA)
        assert np.allclose(np.array(gB._to_nested_list()), expected_gB)


    def test_matmul_backward_batched(self):
        A = cyflow.tensor(np.random.randn(3, 2, 4).tolist(), device=cyflow.CPU, requires_grad=True)
        B = cyflow.tensor(np.random.randn(1, 4, 5).tolist(), device=cyflow.CPU, requires_grad=True)

        C = A @ B
        grad_out = cyflow.tensor(np.ones_like(np.array(C._to_nested_list())).tolist(), device=cyflow.CPU)

        mb = cyflow.autograd.MatmulBackward(A, B)
        gA, gB = mb.apply(grad_out)

        # numerical checks against numpy
        A_np = np.array(A._to_nested_list())
        B_np = np.array(B._to_nested_list())
        grad_np = np.array(grad_out._to_nested_list())

        expected_gA = np.matmul(grad_np, np.swapaxes(B_np, -1, -2))
        expected_gB_full = np.matmul(np.swapaxes(A_np, -1, -2), grad_np)

        # Helper: reduce (sum) broadcasted dims of full -> target
        def unbroadcast_numpy(full, target_shape):
            full_shape = list(full.shape)
            # Pad target_shape on the left with ones
            padded = (1,) * (len(full_shape) - len(target_shape)) + tuple(target_shape)
            res = full
            for ax, (f, t) in enumerate(zip(full_shape, padded)):
                if t == 1 and f != 1:
                    res = res.sum(axis=ax, keepdims=True)
            # finally reshape to target_shape
            return res.reshape(target_shape)

        expected_gB = unbroadcast_numpy(expected_gB_full, B_np.shape)

        assert gA.shape == A.shape
        assert gB.shape == B.shape
        assert np.allclose(np.array(gA._to_nested_list()), expected_gA, atol=1e-6)
        assert np.allclose(np.array(gB._to_nested_list()), expected_gB, atol=1e-6)
