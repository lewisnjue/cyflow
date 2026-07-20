# cython: language_level=3

from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
from cpython.buffer cimport PyBUF_FORMAT, PyBUF_STRIDES, PyBUF_WRITABLE, Py_buffer
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release
from cpython.ref cimport Py_INCREF, Py_DECREF

# =============================================================================
# 1. Declare the C structs and functions from our header
# =============================================================================
cdef extern from "c_tensor.h":
    ctypedef struct Storage:
        float* data
        size_t size
        size_t ref_count
        bint owns_data
        void* owner

    ctypedef struct TensorImpl:
        Storage* storage
        int64_t* shape
        int64_t* strides
        size_t ndim
        size_t storage_offset
        size_t numel

    TensorImpl* tensor_create(const int64_t* shape, size_t ndim)
    TensorImpl* tensor_create_from_buffer(float *data,
                                         const int64_t *shape,
                                         const int64_t *strides,
                                         size_t ndim,
                                         void *owner)
    void tensor_free(TensorImpl* tensor)
    TensorImpl* tensor_matmul(TensorImpl* A, TensorImpl* B)
    TensorImpl *tensor_add(TensorImpl *A, TensorImpl *B);
    TensorImpl *tensor_sub(TensorImpl *A, TensorImpl *B);
    TensorImpl *tensor_mul(TensorImpl *A, TensorImpl *B);
    TensorImpl *tensor_pow(TensorImpl *A, int64_t exponent);
    TensorImpl *tensor_exp(TensorImpl *A);

# =============================================================================
# 2. Python Extension Type
# =============================================================================
cdef class Tensor:
    cdef TensorImpl* _c_tensor

    def __cinit__(self, shape=None):
            if shape is None:
                self._c_tensor = NULL
                return

            # Support direct NumPy-backed tensor creation by accepting an ndarray.
            if hasattr(shape, "__array_interface__") or hasattr(shape, "__array_priority__"):
                self._init_from_numpy(shape)
                return

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

    cdef void _init_from_numpy(self, object np_array):
        cdef Py_buffer view
        cdef int ndim
        cdef int64_t *shape = NULL
        cdef int64_t *strides = NULL
        cdef float *data_ptr
        cdef size_t numel = 1
        cdef size_t i

        if PyObject_GetBuffer(np_array, &view, PyBUF_FORMAT | PyBUF_STRIDES) != 0:
            raise TypeError("Unable to get buffer from NumPy array")

        try:
            if view.ndim < 1:
                raise ValueError("NumPy array must have at least 1 dimension")
            if view.len == 0:
                raise ValueError("NumPy array must be non-empty")
            if view.itemsize != sizeof(float):
                raise TypeError("NumPy array must have dtype float32")
            if view.format[0] != 'f' and strcmp(view.format, "f") != 0:
                raise TypeError("NumPy array must have dtype float32")

            ndim = view.ndim
            shape = <int64_t*>malloc(ndim * sizeof(int64_t))
            strides = <int64_t*>malloc(ndim * sizeof(int64_t))
            if not shape or not strides:
                if shape:
                    free(shape)
                if strides:
                    free(strides)
                raise MemoryError("Failed to allocate shape or strides array")

            for i in range(ndim):
                shape[i] = view.shape[i]
                strides[i] = view.strides[i] // view.itemsize
                numel *= <size_t>shape[i]

            if view.suboffsets:
                for i in range(ndim):
                    if view.suboffsets[i] != -1:
                        free(shape)
                        free(strides)
                        raise TypeError("NumPy array must be contiguous or strided without suboffsets")

            data_ptr = <float*>view.buf
            self._c_tensor = tensor_create_from_buffer(data_ptr, shape, strides, ndim, <void*>np_array)
        finally:
            PyBuffer_Release(&view)

        if self._c_tensor == NULL:
            raise MemoryError("Failed to create Tensor from NumPy buffer")

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
        return f"<cyflow.Tensor shape={self.shape} strides={self.strides} numel={self.numel}>"
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented

        cdef Tensor other_tensor = <Tensor>other

        # Call our C engine
        cdef TensorImpl* result_impl = tensor_matmul(self._c_tensor, other_tensor._c_tensor)

        if result_impl == NULL:
            raise ValueError("Matrix multiplication failed. Check if shapes are compatible.")

        # Wrap the C pointer in a new Python Tensor object without allocating new memory
        cdef Tensor result = Tensor.__new__(Tensor) # Calls __cinit__(shape=None)
        result._c_tensor = result_impl

        return result
    
    def __add__(self,other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_add(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Addition failed. Check if shapes are compatible.")
        cdef Tensor result = Tensor.__new__(Tensor)
        result._c_tensor = result_impl
        return result
    
    def __sub__(self,other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_sub(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Subtraction failed. Check if shapes are compatible.")
        cdef Tensor result = Tensor.__new__(Tensor)
        result._c_tensor = result_impl
        return result
    
    def __mul__(self,other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_mul(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Multiplication failed. Check if shapes are compatible.")
        cdef Tensor result = Tensor.__new__(Tensor)
        result._c_tensor = result_impl
        return result

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        cdef TensorImpl* result_impl = tensor_pow(self._c_tensor, exponent)
        if result_impl == NULL:
            raise ValueError("Power operation failed.")
        cdef Tensor result = Tensor.__new__(Tensor)
        result._c_tensor = result_impl
        return result
    
    def exp(self):
        cdef TensorImpl* result_impl = tensor_exp(self._c_tensor)
        if result_impl == NULL:
            raise ValueError("Exponential operation failed.")
        cdef Tensor result = Tensor.__new__(Tensor)
        result._c_tensor = result_impl
        return result
    