import pytest
from cyflow import tensor, Tensor, CPU


class TestBackward:
    """Test the backward() method for computing gradients."""

    def test_backward_scalar_addition(self):
        """Test backward through scalar addition (y = x + 2, backward from y[0])."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x + 2.0
        
        # Get scalar element and call backward
        scalar_output = y[0]
        scalar_output.backward()
        
        # Gradient should exist for x
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_scalar_subtraction(self):
        """Test backward through scalar subtraction (y = x - 1)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x - 1.0
        
        scalar_output = y[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_scalar_multiplication(self):
        """Test backward through scalar multiplication (y = x * 3)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 3.0
        
        scalar_output = y[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_scalar_division(self):
        """Test backward through scalar division (y = x / 2)."""
        x = tensor([2.0, 4.0, 6.0], requires_grad=True)
        y = x / 2.0
        
        scalar_output = y[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_tensor_addition(self):
        """Test backward through tensor addition (z = x + y)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = tensor([0.5, 0.5, 0.5], requires_grad=True)
        z = x + y
        
        scalar_output = z[0]
        scalar_output.backward()
        
        # Both x and y should have gradients
        assert x.grad is not None
        assert y.grad is not None

    def test_backward_tensor_subtraction(self):
        """Test backward through tensor subtraction (z = x - y)."""
        x = tensor([3.0, 4.0, 5.0], requires_grad=True)
        y = tensor([1.0, 1.0, 1.0], requires_grad=True)
        z = x - y
        
        scalar_output = z[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert y.grad is not None

    def test_backward_tensor_multiplication(self):
        """Test backward through tensor multiplication (z = x * y)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = tensor([2.0, 2.0, 2.0], requires_grad=True)
        z = x * y
        
        scalar_output = z[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert y.grad is not None

    def test_backward_tensor_division(self):
        """Test backward through tensor division (z = x / y)."""
        x = tensor([2.0, 4.0, 6.0], requires_grad=True)
        y = tensor([2.0, 2.0, 2.0], requires_grad=True)
        z = x / y
        
        scalar_output = z[0]
        scalar_output.backward()
        
        assert x.grad is not None
        assert y.grad is not None

    def test_backward_chain_rule_add_mul(self):
        """Test backward through chained operations (z = (x + 2) * 3)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x + 2.0
        z = y * 3.0
        
        scalar_output = z[0]
        scalar_output.backward()
        
        # x should have gradient from chain rule
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_chain_rule_mul_sub(self):
        """Test backward through (z = (x * 2) - 1)."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2.0
        z = y - 1.0
        
        scalar_output = z[0]
        scalar_output.backward()
        
        assert x.grad is not None

    def test_backward_requires_grad_false(self):
        """Test that backward raises error on tensor without requires_grad."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=False)
        y = x + 1.0
        
        with pytest.raises(RuntimeError):
            y.backward()

    def test_backward_non_scalar_raises(self):
        """Test that backward on non-scalar without explicit gradient raises."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x + 1.0
        
        with pytest.raises(RuntimeError, match="scalar"):
            y.backward()  # y is not a scalar

    def test_backward_with_explicit_gradient(self):
        """Test backward with an explicit gradient provided."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2.0
        
        # Create explicit gradient tensor
        grad = tensor([1.0, 1.0, 1.0], requires_grad=False)
        y.backward(gradient=grad)
        
        assert x.grad is not None

    def test_backward_leaf_tensor_error(self):
        """Test that calling backward on a leaf tensor without operations raises."""
        x = tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        with pytest.raises(RuntimeError, match="Tensor is a leaf"):
            x.backward()
