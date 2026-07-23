import pytest
import cyflow
from cyflow import Tensor, CPU, CUDA, manual_seed, tensor

def _get_active_devices():
    """Detects available devices: always runs CPU, adds CUDA if available."""
    devices = [cyflow.CPU]
    try:
        # Attempt to initialize a tiny CUDA tensor to test hardware availability
        cyflow.tensor(shape=(1,), device=cyflow.CUDA)
        devices.append(cyflow.CUDA)
    except Exception:
        pass
    return devices

DEVICES = _get_active_devices()

@pytest.mark.parametrize("device", DEVICES)
class TestCyflowTensor:

    def test_tensor_creation_and_shape_normalization(self, device):
        # Test tuple shape
        t1 = cyflow.tensor(shape=(2, 3), device=device)
        assert t1.shape == (2, 3)
        assert t1.ndim == 2
        assert t1.numel == 6
        assert t1.nbytes == 6 * 4

        # Test integer shape normalization e.g. cyflow.tensor(5) -> (5,)
        t2 = cyflow.tensor(shape=5, device=device)
        assert t2.shape == (5,)
        assert t2.ndim == 1
        assert t2.numel == 5

    def test_tensor_properties(self, device):
        t = cyflow.tensor(shape=(2, 2), device=device)
        assert t.device == ("cpu" if device == cyflow.CPU else "cuda")
        assert isinstance(t.strides, tuple)
        assert len(t.strides) == t.ndim

    def test_item_and_scalar_assignment(self, device):
        t = cyflow.tensor(shape=(1,), device=device)
        t[:] = 42.0
        assert t.item() == 42.0

    def test_list_assignment_and_nested_list(self, device):
        t = cyflow.tensor(shape=(2, 2), device=device)
        data = [[1.0, 2.0], [3.0, 4.0]]
        t[:] = data
        assert t._to_nested_list() == data

    def test_manual_seed_and_fill_uniform(self, device):
        cyflow.manual_seed(123, device=device)
        t = cyflow.tensor(shape=(3, 3), device=device)
        t.fill_uniform()
        assert t.numel == 9
        nested = t._to_nested_list()
        assert len(nested) == 3

    def test_inplace_scalar_arithmetic(self, device):
        t = cyflow.tensor(shape=(2, 2), device=device)
        t[:] = 4.0
        
        t += 2.0
        assert t._to_nested_list() == [[6.0, 6.0], [6.0, 6.0]]

        t -= 1.0
        assert t._to_nested_list() == [[5.0, 5.0], [5.0, 5.0]]

        t *= 3.0
        assert t._to_nested_list() == [[15.0, 15.0], [15.0, 15.0]]

        t /= 5.0
        assert t._to_nested_list() == [[3.0, 3.0], [3.0, 3.0]]

    def test_inplace_tensor_arithmetic(self, device):
        t1 = cyflow.tensor(shape=(2, 2), device=device)
        t2 = cyflow.tensor(shape=(2, 2), device=device)
        t1[:] = 6.0
        t2[:] = 2.0

        t1 += t2
        assert t1._to_nested_list() == [[8.0, 8.0], [8.0, 8.0]]

        t1 -= t2
        assert t1._to_nested_list() == [[6.0, 6.0], [6.0, 6.0]]

        t1 *= t2
        assert t1._to_nested_list() == [[12.0, 12.0], [12.0, 12.0]]

        t1 /= t2
        assert t1._to_nested_list() == [[6.0, 6.0], [6.0, 6.0]]

    def test_indexing_and_slicing(self, device):
        t = cyflow.tensor(shape=(3, 3), device=device)
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        t[:] = data

        # Basic item indexing
        assert t[0, 0].item() == 1.0
        assert t[2, 1].item() == 8.0

        # Slicing
        sub = t[0:2, 1:3]
        assert sub.shape == (2, 2)
        assert sub._to_nested_list() == [[2.0, 3.0], [5.0, 6.0]]

        # Scalar slice assignment
        t[0:2, 0:2] = 0.0
        assert t[0, 0].item() == 0.0
        assert t[1, 1].item() == 0.0

        # Tensor slice assignment
        src = cyflow.tensor(shape=(2, 2), device=device)
        src[:] = 99.0
        t[1:3, 1:3] = src
        assert t[1, 1].item() == 99.0
        assert t[2, 2].item() == 99.0

    def test_tensor_view(self, device):
        t = cyflow.tensor(shape=(2, 3), device=device)
        t[:] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        v = t.view(3, 2)
        assert v.shape == (3, 2)
        assert v.numel == 6

        with pytest.raises(ValueError):
            t.view(5)  # Mismatched element count should raise ValueError