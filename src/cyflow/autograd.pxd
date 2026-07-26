from cyflow.tensor cimport Tensor

cdef class AutogradNode:
    cdef public list next_functions
    cpdef tuple apply(self, Tensor grad_output)

cdef class AddBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public object other
    cpdef tuple apply(self, Tensor grad_output)

cdef class SubBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public object other
    cpdef tuple apply(self, Tensor grad_output)

cdef class MulBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public object other
    cpdef tuple apply(self, Tensor grad_output)

cdef class DivBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public object other
    cpdef tuple apply(self, Tensor grad_output)

cdef class ViewBackward(AutogradNode):
    cdef public Tensor self_tensor
    cpdef tuple apply(self, Tensor grad_output)

cdef class GetItemBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public Tensor output_view
    cpdef tuple apply(self, Tensor grad_output)
