from cyflow.tensor cimport Tensor
from cyflow.tensor cimport unbroadcast, accumulate_view_grad
from libc.stdlib cimport malloc, free, calloc
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libc.string cimport memcpy

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


cdef class SubBackward(AutogradNode):
    def __init__(self, Tensor self_tensor, object other):
        super().__init__()
        self.self_tensor = self_tensor
        self.other = other

        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)

        if not isinstance(self.other, (int, float)):
            if (<Tensor>self.other).requires_grad:
                self.next_functions.append((<Tensor>self.other).grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_self = None
        cdef Tensor grad_other = None
        cdef Tensor other_t
        cdef Tensor neg_grad

        # 1. Gradient for `self_tensor`
        if self.self_tensor.requires_grad:
            if self.self_tensor.shape != grad_output.shape:
                grad_self = unbroadcast(grad_output, self.self_tensor.shape)
            else:
                grad_self = grad_output

        # 2. Gradient for `other`
        if not isinstance(self.other, (int, float)):
            other_t = <Tensor>self.other
            if other_t.requires_grad:
                neg_grad = grad_output.detach() * -1.0
                if other_t.shape != neg_grad.shape:
                    grad_other = unbroadcast(neg_grad, other_t.shape)
                    del neg_grad
                else:
                    grad_other = neg_grad

        return grad_self, grad_other


cdef class MulBackward(AutogradNode):
    def __init__(self, Tensor self_tensor, object other):
        super().__init__()
        self.self_tensor = self_tensor
        self.other = other

        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)

        if not isinstance(self.other, (int, float)):
            if (<Tensor>self.other).requires_grad:
                self.next_functions.append((<Tensor>self.other).grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_self = None
        cdef Tensor grad_other = None
        cdef Tensor other_t, detached_grad_out, detached_other, detached_self
        cdef Tensor temp_self, temp_other

        detached_grad_out = grad_output.detach()

        # 1. Gradient for `self_tensor` (grad_output * other)
        if self.self_tensor.requires_grad:
            if isinstance(self.other, (int, float)):
                temp_self = detached_grad_out * self.other
            else:
                other_t = <Tensor>self.other
                detached_other = other_t.detach()
                temp_self = detached_grad_out * detached_other
                del detached_other

            if self.self_tensor.shape != temp_self.shape:
                grad_self = unbroadcast(temp_self, self.self_tensor.shape)
                del temp_self
            else:
                grad_self = temp_self

        # 2. Gradient for `other` (grad_output * self)
        if not isinstance(self.other, (int, float)):
            other_t = <Tensor>self.other
            if other_t.requires_grad:
                detached_self = self.self_tensor.detach()
                temp_other = detached_grad_out * detached_self
                del detached_self

                if other_t.shape != temp_other.shape:
                    grad_other = unbroadcast(temp_other, other_t.shape)
                    del temp_other
                else:
                    grad_other = temp_other

        del detached_grad_out
        return grad_self, grad_other


cdef class DivBackward(AutogradNode):
    def __init__(self, Tensor self_tensor, object other):
        super().__init__()
        self.self_tensor = self_tensor
        self.other = other

        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)

        if not isinstance(self.other, (int, float)):
            if (<Tensor>self.other).requires_grad:
                self.next_functions.append((<Tensor>self.other).grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_self = None
        cdef Tensor grad_other = None
        cdef Tensor other_t, detached_grad_out, detached_other, detached_self
        cdef Tensor temp_self, temp_other, temp1, temp2, temp3

        detached_grad_out = grad_output.detach()

        # 1. Gradient for `self_tensor` (grad_output / other)
        if self.self_tensor.requires_grad:
            if isinstance(self.other, (int, float)):
                temp_self = detached_grad_out / self.other
            else:
                other_t = <Tensor>self.other
                detached_other = other_t.detach()
                temp_self = detached_grad_out / detached_other
                del detached_other

            if self.self_tensor.shape != temp_self.shape:
                grad_self = unbroadcast(temp_self, self.self_tensor.shape)
                del temp_self
            else:
                grad_self = temp_self

        # 2. Gradient for `other` (-grad_output * self / (other * other))
        if not isinstance(self.other, (int, float)):
            other_t = <Tensor>self.other
            if other_t.requires_grad:
                detached_self = self.self_tensor.detach()
                detached_other = other_t.detach()

                temp1 = detached_grad_out * -1.0
                temp2 = temp1 * detached_self
                temp3 = detached_other * detached_other
                temp_other = temp2 / temp3

                # Clean up intermediate steps
                del detached_self
                del detached_other
                del temp1
                del temp2
                del temp3

                if other_t.shape != temp_other.shape:
                    grad_other = unbroadcast(temp_other, other_t.shape)
                    del temp_other
                else:
                    grad_other = temp_other

        del detached_grad_out
        return grad_self, grad_other


cdef class ViewBackward(AutogradNode):
    def __init__(self, Tensor input_tensor):
        super().__init__()
        self.self_tensor = input_tensor

        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_input = None
        cdef Tensor contig_grad = grad_output

        if self.self_tensor.requires_grad:
            # Force contiguous layout if strided
            if not grad_output.is_contiguous():
                # detach() uses tensor_clone_cpu/cuda which packs data into a contiguous buffer
                contig_grad = grad_output.detach()

            grad_input = contig_grad.view(self.self_tensor.shape)

        return grad_input, None


cdef class GetItemBackward(AutogradNode):
    def __init__(self, Tensor input_tensor, Tensor output_view):
        super().__init__()
        self.self_tensor = input_tensor
        self.output_view = output_view

        if self.self_tensor.requires_grad:
            self.next_functions.append(self.self_tensor.grad_fn)

    cpdef tuple apply(self, Tensor grad_output):
        cdef Tensor grad_input = None

        if self.self_tensor.requires_grad:
            grad_input = accumulate_view_grad(
                self.self_tensor,
                self.output_view,
                grad_output,
            )

        return grad_input, None
