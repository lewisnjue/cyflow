import pytest
import cyflow



class TestAddScaler:
    def setup_method(self):
        self.t = cyflow.tensor([[1, 2], [3, 4]])

    def test_add_scalar_cpu(self):
        result = self.t + 5
        assert result.shape == (2, 2)
        assert result.numel == 4
        assert result.device == "cpu"


class TestAddTensor:
    def setup_method(self):
        self.t1 = cyflow.tensor([[1, 2], [3, 4]])
        self.t2 = cyflow.tensor([[5, 6], [7, 8]])

    def test_add_tensor_cpu(self):
        result = self.t1 + self.t2
        assert result.shape == (2, 2)
        assert result.numel == 4
        assert result.device == "cpu"

    
    def test_add_tensor_device_mismatch(self):
        t1 = cyflow.tensor([[1, 2], [3, 4]], device=cyflow.CPU)
        t2 = cyflow.tensor([[5, 6], [7, 8]], device=cyflow.CUDA)
        with pytest.raises(AssertionError):
            _ = t1 + t2

    def test_add_broadcasting(self):
        t1 = cyflow.tensor([[1, 2], [3, 4]])
        t2 = cyflow.tensor([10, 20])
        result = t1 + t2
        assert result.shape == (2, 2)
        assert result.numel == 4
        assert result.device == "cpu"