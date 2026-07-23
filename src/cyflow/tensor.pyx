# cython: language_level=3

from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

cdef extern from "cyflow/common.h":
    ctypedef enum DeviceType:
        DEVICE_CPU
        DEVICE_CUDA

    ctypedef struct Storage:
        float *data
        size_t size
        int ref_count
        bint owns_data
        DeviceType device

    ctypedef struct TensorImpl:
        Storage *storage
        int64_t *shape
        int64_t *strides
        size_t ndim
        size_t numel
        size_t storage_offset # Fixed: matches the C header exactly

    void compute_contiguous_strides(int64_t *strides, const int64_t *shape, size_t ndim)
    TensorImpl *tensor_view(TensorImpl *src, const int64_t *shape, size_t ndim)


cdef extern from "cyflow/tensor_cpu.h":
    Storage *storage_create_cpu(size_t size)
    void storage_free_cpu(Storage *storage)
    TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim)
    void tensor_free_cpu(TensorImpl *tensor)
    void cyflow_manual_seed(unsigned int seed)
    void tensor_fill_uniform_cpu(TensorImpl *tensor)
    void tensor_set_data_cpu(TensorImpl *tensor, const float *data)

cdef extern from "cyflow/tensor_cuda.h":
    Storage *storage_create_cuda(size_t size)
    void storage_free_cuda(Storage *storage)
    TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim)
    void tensor_free_cuda(TensorImpl *tensor)
    void tensor_fill_uniform_cuda(TensorImpl *tensor)
    void cyflow_manual_seed_cuda(unsigned long long seed)
    void tensor_set_data_cuda(TensorImpl *tensor, const float *data)

CPU = DEVICE_CPU
CUDA = DEVICE_CUDA

cpdef manual_seed(unsigned long long seed, int device=CPU):
    if device == DEVICE_CPU:
        cyflow_manual_seed(seed)
    elif device == DEVICE_CUDA:
        cyflow_manual_seed_cuda(seed)
    else:
        raise ValueError(f"Unsupported device integer: {device}")

cdef class Tensor:
    cdef TensorImpl* _tensor

    def __cinit__(self, shape = None, int device=CPU):
        if shape is None:
            self._tensor = NULL
            return
        cdef size_t ndim = len(shape)
        cdef int64_t* c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
        if not c_shape:
            raise MemoryError("Failed to allocate shape array")

        for i in range(ndim):
            c_shape[i] = shape[i]

        try:
            if device == DEVICE_CPU:
                self._tensor = tensor_create_cpu(c_shape, ndim)
            elif device == DEVICE_CUDA:
                self._tensor = tensor_create_cuda(c_shape, ndim)
            else:
                raise ValueError(f"Unsupported device integer: {device}")

            if self._tensor is NULL:
                raise MemoryError("Backend failed to allocate TensorImpl")
        finally:
            free(c_shape)

    def __dealloc__(self):
        if self._tensor is not NULL:
            if self._tensor.storage.device == DEVICE_CPU:
                tensor_free_cpu(self._tensor)
            elif self._tensor.storage.device == DEVICE_CUDA:
                tensor_free_cuda(self._tensor)
    @staticmethod
    cdef Tensor _from_c_tensor(TensorImpl* ptr):
        cdef Tensor t = Tensor.__new__(Tensor)
        t._tensor = ptr
        return t

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def numel(self):
        return self._tensor.numel

    @property
    def shape(self):
        return tuple([self._tensor.shape[i] for i in range(self._tensor.ndim)])

    @property
    def strides(self):
        return tuple([self._tensor.strides[i] for i in range(self._tensor.ndim)])

    @property
    def device(self):
        if self._tensor.storage.device == DEVICE_CPU:
            return "cpu"
        elif self._tensor.storage.device == DEVICE_CUDA:
            return "cuda"
        return "unknown"

    def __repr__(self):
        return f"<Tensor shape={self.shape} strides={self.strides} device='{self.device}'>"
    @property
    def nbytes(self):
        return self.numel * sizeof(float)

    def fill_uniform(self):
            if self._tensor.storage.device == DEVICE_CPU:
                tensor_fill_uniform_cpu(self._tensor)
            elif self._tensor.storage.device == DEVICE_CUDA:
                tensor_fill_uniform_cuda(self._tensor)
    def view(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        cdef size_t target_numel = 1
        for dim in target_shape:
            target_numel *= dim

        if target_numel != self.numel:
            raise ValueError(f"Cannot reshape tensor of size {self.numel} into shape {target_shape}")

        cdef size_t ndim = len(target_shape)
        cdef int64_t* c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
        if not c_shape:
            raise MemoryError("Failed to allocate memory for shape array")

        for i in range(ndim):
            c_shape[i] = target_shape[i]

        cdef TensorImpl* new_impl = NULL
        try:
            new_impl = tensor_view(self._tensor, c_shape, ndim)
        finally:
            free(c_shape)

        if new_impl is NULL:
            raise RuntimeError("Backend failed to create tensor view")

        return Tensor._from_c_tensor(new_impl)

    def _set_data_from_list(self, flat_data: list):
                if len(flat_data) != self.numel:
                    raise ValueError(f"Expected {self.numel} elements, got {len(flat_data)}")

                cdef size_t numel = self.numel
                cdef float* c_data = <float*>malloc(numel * sizeof(float))
                if not c_data:
                    raise MemoryError("Failed to allocate temporary data buffer")

                try:
                    for i in range(numel):
                        c_data[i] = float(flat_data[i])

                    if self._tensor.storage.device == DEVICE_CPU:
                        tensor_set_data_cpu(self._tensor, c_data)
                    elif self._tensor.storage.device == DEVICE_CUDA:
                        tensor_set_data_cuda(self._tensor, c_data)
                finally:
                    free(c_data)
