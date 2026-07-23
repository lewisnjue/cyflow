import pytest

from cyflow import tensor, Tensor, CPU, manual_seed


class TestCreation:
    def test_from_nested_list(self):
        t = tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.numel == 4
        assert t.device == "cpu"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            tensor([[1, 2], [3, 4]], shape=(3,))

    def test_create_from_shape_fills(self):
        manual_seed(123, device=CPU)
        t = tensor(shape=(2, 3), device=CPU)
        assert t.shape == (2, 3)
        assert t.numel == 6


class TestGetItem:
    def setup_method(self):
        self.t = tensor([[1, 2, 3], [4, 5, 6]])

    def test_getitem_int(self):
        r = self.t[0]
        assert isinstance(r, Tensor)
        assert r.shape == (3,)
        assert r.numel == 3

    def test_getitem_slice(self):
        r = self.t[:, 1]
        assert r.shape == (2,)
        assert r.numel == 2

    def test_getitem_ellipsis_and_newaxis(self):
        r = self.t[..., None]
        assert r.shape == (2, 3, 1)
        assert r.numel == 6

    def test_getitem_list_index_not_supported(self):
        with pytest.raises(NotImplementedError):
            _ = self.t[[0, 1]]

    def test_too_many_indices(self):
        with pytest.raises(IndexError):
            _ = self.t[0, 0, 0]

    def test_negative_index(self):
        r = self.t[-1]
        assert r.shape == (3,)


class TestSetItem:
    def setup_method(self):
        self.t = tensor([[0, 0], [0, 0]])

    def test_setitem_scalar(self):
        # assigning scalar should not raise and shape remains
        self.t[0, 0] = 5
        assert self.t.shape == (2, 2)

    def test_setitem_list(self):
        # assigning a compatible list to a slice should not raise
        self.t[0] = [1, 2]
        assert self.t.shape == (2, 2)

    def test_setitem_list_shape_mismatch(self):
        with pytest.raises(ValueError):
            self.t[0] = [1, 2, 3]

    def test_setitem_tensor_shape_mismatch_raises(self):
        other = tensor([1, 2, 3])
        with pytest.raises(ValueError):
            self.t[0] = other

    def test_setitem_invalid_type(self):
        with pytest.raises(TypeError):
            self.t[0] = "abc"

    def test_view_reshape_success_and_failure(self):
        t = tensor([[1, 2], [3, 4]])
        v = t.view(4)
        assert v.shape == (4,)
        assert v.numel == 4
        with pytest.raises(ValueError):
            t.view(3)
