# cython: language_level=3

from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

# =============================================================================
# 1. Declare the C structs and functions from our header
# =============================================================================
cdef extern from "c_tensor.h":
    ctypedef struct Storage:
        float* data
        size_t size
        size_t ref_count

    ctypedef struct TensorImpl:
        Storage* storage
        int64_t* shape
        int64_t* strides
        size_t ndim
        size_t storage_offset
        size_t numel

    TensorImpl* tensor_create(const int64_t* shape, size_t ndim)
    void tensor_free(TensorImpl* tensor)

# =============================================================================
# 2. Python Extension Type
# =============================================================================
cdef class Tensor:
    cdef TensorImpl* _c_tensor

    def __cinit__(self, shape: tuple):
        cdef size_t ndim = len(shape)
        cdef int64_t* c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
        if not c_shape:
            raise MemoryError("Failed to allocate shape array")

        for i in range(ndim):
            c_shape[i] = shape[i]

        self._c_tensor = tensor_create(c_shape, ndim)
        free(c_shape)

        if self._c_tensor == NULL:
            raise MemoryError("Failed to allocate C Tensor")

    def __dealloc__(self):
        if self._c_tensor != NULL:
            tensor_free(self._c_tensor)
            self._c_tensor = NULL

    # --- Indexing & Memory Access ---

    cdef size_t _get_flat_index(self, index) except *:
        cdef size_t flat_idx = self._c_tensor.storage_offset
        cdef size_t i
        cdef int64_t idx

        if isinstance(index, int):
            if self.ndim != 1:
                raise IndexError(f"Expected tuple of length {self.ndim}, got int")
            index = (index,)

        if not isinstance(index, tuple):
            raise TypeError("Index must be an int or tuple of ints")

        if len(index) != self.ndim:
            raise IndexError(f"Expected index of length {self.ndim}, got {len(index)}")

        for i in range(self.ndim):
            idx = index[i]

            if idx < 0:
                idx += self._c_tensor.shape[i]

            if idx < 0 or idx >= self._c_tensor.shape[i]:
                raise IndexError(f"Index {index[i]} is out of bounds for axis {i} with size {self._c_tensor.shape[i]}")

            flat_idx += idx * self._c_tensor.strides[i]

        return flat_idx

    def __getitem__(self, index):
        """Read a float from the tensor."""
        cdef size_t flat_idx = self._get_flat_index(index)
        return self._c_tensor.storage.data[flat_idx]

    def __setitem__(self, index, value):
        """Write a float to the tensor."""
        cdef float c_value = value
        cdef size_t flat_idx = self._get_flat_index(index)
        self._c_tensor.storage.data[flat_idx] = c_value

    # --- Python Properties ---

    @property
    def shape(self):
        return tuple(self._c_tensor.shape[i] for i in range(self._c_tensor.ndim))

    @property
    def strides(self):
        return tuple(self._c_tensor.strides[i] for i in range(self._c_tensor.ndim))

    @property
    def ndim(self):
        return self._c_tensor.ndim

    @property
    def numel(self):
        return self._c_tensor.numel

    def __repr__(self):
        return f"<nnetflow.Tensor shape={self.shape} strides={self.strides} numel={self.numel}>"
