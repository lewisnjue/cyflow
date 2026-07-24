from cyflow.tensor cimport Tensor
from cyflow.tensor cimport unbroadcast

cdef class AutogradNode:
    def __init__(self):
        self.next_functions = []

    cpdef tuple apply(self, Tensor grad_output):
        raise NotImplementedError("Must be implemented by subclasses")


cdef class AddBackward(AutogradNode):
    def __init__(self, Tensor self_tensor, object other):
        super().__init__()
        self.self_tensor = self_tensor
        self.other = other
        
        # Build graph edges. If a tensor is a leaf (created by user), its grad_fn is None.
        # We append None so the engine knows it reached a leaf and should accumulate the gradient.
        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)
            
        if not isinstance(self.other, (int, float)):
            if (<Tensor>self.other).requires_grad:
                self.next_functions.append((<Tensor>self.other).grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_self = None
        cdef Tensor grad_other = None
        cdef Tensor other_t
        
        # 1. Gradient for `self_tensor`
        if self.self_tensor.requires_grad:
            if self.self_tensor.shape != grad_output.shape:
                # Unbroadcast sum for self
                grad_self = unbroadcast(grad_output, self.self_tensor.shape)
            else:
                # Direct pass-through
                grad_self = grad_output

        # 2. Gradient for `other` (only if it's a Tensor)
        if not isinstance(self.other, (int, float)):
            other_t = <Tensor>self.other
            if other_t.requires_grad:
                if other_t.shape != grad_output.shape:
                    # Unbroadcast sum for other
                    grad_other = unbroadcast(grad_output, other_t.shape)
                else:
                    grad_other = grad_output

        return grad_self, grad_other