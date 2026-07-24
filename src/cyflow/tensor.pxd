from libc.stdint cimport int64_t
from libc.stddef cimport size_t
cdef extern from "cyflow/tensor.h":
    ctypedef struct TensorImpl:
        Storage *storage
        int64_t *shape
        int64_t *strides
        size_t ndim
        size_t numel
        size_t storage_offset

   

    ctypedef struct Storage:
            float *data
            size_t size
            int ref_count
            bint owns_data
            DeviceType device

    ctypedef enum DeviceType:
            DEVICE_CPU
            DEVICE_CUDA

cdef class Tensor:
    cdef TensorImpl* _tensor
    cdef public bint requires_grad
    cdef public object grad
    cdef public object grad_fn

    @staticmethod
    cdef Tensor _from_c_tensor(TensorImpl* ptr)

    cpdef _to_nested_list(self)
    cdef _fill_scalar(self, float val)
    cdef _fill_from_flat_list(self, list flat_vals)
    cdef _copy_from_tensor(self, Tensor src)
    cpdef _apply_inplace(self, object other, str op)


cpdef Tensor unbroadcast(Tensor grad, tuple target_shape)