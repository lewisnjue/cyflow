from pathlib import Path
import sys
from cyflow import Tensor

# Keep the path manipulation at the top so pytest can find the source code
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_tensor_shape_and_metadata() -> None:
    tensor = Tensor((2, 3))

    assert tensor.shape == (2, 3)
    assert tensor.strides == (3, 1)
    assert tensor.ndim == 2
    assert tensor.numel == 6


def test_tensor_repr_contains_shape_and_numel() -> None:
    tensor = Tensor((1, 4))
    representation = repr(tensor)

    assert "shape=(1, 4)" in representation
    assert "numel=4" in representation


def test_tensor_can_be_imported_from_package_root() -> None:
    from cyflow import Tensor as PackageTensor

    assert PackageTensor is Tensor

