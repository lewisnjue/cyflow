from cyflow.tensor cimport Tensor

# 1. Base Node
cdef class AutogradNode:
    cdef list next_functions # A list of nodes to pass gradients to

    def apply(self, Tensor grad_output):
        raise NotImplementedError("Must be implemented by subclasses")


# 2. Addition Backward Node
cdef class AddBackward(AutogradNode):
    # We must save references to the input tensors to pass gradients to them
    cdef Tensor saved_self
    cdef Tensor saved_other

    def __init__(self, Tensor self_tensor, Tensor other_tensor):
        self.next_functions = []
        self.saved_self = self_tensor
        self.saved_other = other_tensor

        # Build the graph edges: if the input tensor was created by an operation,
        # we append its grad_fn. Otherwise, it's a leaf tensor, and we will
        # accumulate the gradient directly into it later.
        if self_tensor.grad_fn is not None:
            self.next_functions.append(self_tensor.grad_fn)
        if other_tensor.grad_fn is not None:
            self.next_functions.append(other_tensor.grad_fn)

    def apply(self, Tensor grad_output):
        """
        The calculus of Addition:
        If z = x + y
        Then dz/dx = 1  -> Therefore, gradient of x = grad_output * 1
        And  dz/dy = 1  -> Therefore, gradient of y = grad_output * 1
        """

        # In addition, the gradient flows backwards equally to both inputs.
        # grad_self  = grad_output * 1
        # grad_other = grad_output * 1

        # NOTE: We return two gradients because this operation took two inputs.
        # We will implement the actual tensor engine traversing later, but for now,
        # it just returns the gradients it calculated.

        return grad_output, grad_output
