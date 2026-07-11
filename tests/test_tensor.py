from nnetflow import Tensor
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TensorTests(unittest.TestCase):
    def test_tensor_shape_and_metadata(self) -> None:
        tensor = Tensor((2, 3))

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.strides, (3, 1))
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.numel, 6)

    def test_tensor_repr_contains_shape_and_numel(self) -> None:
        tensor = Tensor((1, 4))
        representation = repr(tensor)

        self.assertIn("shape=(1, 4)", representation)
        self.assertIn("numel=4", representation)

    def test_tensor_can_be_imported_from_package_root(self) -> None:
        from nnetflow import Tensor as PackageTensor

        self.assertIs(PackageTensor, Tensor)


if __name__ == "__main__":
    unittest.main()
