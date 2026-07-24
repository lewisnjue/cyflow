from cyflow.tensor cimport Tensor

cdef class AutogradNode:
    cdef public list next_functions
    cpdef tuple apply(self, Tensor grad_output)

cdef class AddBackward(AutogradNode):
    cdef public Tensor self_tensor
    cdef public object other
    cpdef tuple apply(self, Tensor grad_output)