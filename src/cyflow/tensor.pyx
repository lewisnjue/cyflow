# cython: language_level=3
from typing import Tuple, Set, Optional
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from libc.string cimport strcmp
from cpython.buffer cimport PyBUF_FORMAT, PyBUF_STRIDES, PyBUF_WRITABLE, Py_buffer
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release
from cpython.ref cimport Py_INCREF, Py_DECREF

# Initialize numpy's C-API to avoid runtime errors when integrating cython and numpy
np.import_array()

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
    cdef public set _prev
    cdef public str _op
    cdef public bint requires_grad
    cdef public object grad
    cdef public object _backward

    def __cinit__(self, shape=None, requires_grad: Optional[bool]=None, _children: Tuple['Tensor',...] = (), _op: str = ''):
        cdef size_t ndim
        cdef int64_t* c_shape
        cdef size_t i

        if shape is None:
            self._c_tensor = NULL
        else:
            # Support direct NumPy-backed tensor creation by accepting an ndarray.
            if hasattr(shape, "__array_interface__") or hasattr(shape, "__array_priority__"):
                self._init_from_numpy(shape)
            else:
                ndim = len(shape)
                c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
                if not c_shape:
                    raise MemoryError("Failed to allocate shape array")

                for i in range(ndim):
                    c_shape[i] = shape[i]

                self._c_tensor = tensor_create(c_shape, ndim)
                free(c_shape)

                if self._c_tensor == NULL:
                    raise MemoryError("Failed to allocate C Tensor")

        # Autograd Graph Attributes
        self._prev = set(c for c in _children if isinstance(c, Tensor))
        self._op = _op
        
        if requires_grad is None:
            self.requires_grad = any(c.requires_grad for c in self._prev)
        else:
            self.requires_grad = bool(requires_grad)
            
        self.grad = None 
        if self.requires_grad and self._c_tensor != NULL:
            self.grad = np.zeros(self.shape, dtype=np.float32)

        self._backward = lambda: None # Placeholder for backward function
        
    def zero_grad(self):
        """ Reset the gradient of the tensor to zero. """
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros(self.shape, dtype=np.float32)
            else:
                self.grad.fill(0.0)

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
            if view.format[0] != b'f' and strcmp(view.format, b"f") != 0:
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

    # --- Buffer Protocol for Zero-Copy NumPy Views ---

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """Allows seamlessly converting the Cython Tensor to a NumPy array."""
        if self._c_tensor == NULL:
            raise RuntimeError("Cannot get buffer from uninitialized Tensor")

        cdef int itemsize = sizeof(float)
        
        # Position pointer with respect to storage_offset
        buffer.buf = <char *>(self._c_tensor.storage.data) + self._c_tensor.storage_offset * itemsize
        buffer.format = b'f'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.numel * itemsize
        buffer.ndim = self.ndim
        buffer.obj = self
        buffer.readonly = 0

        buffer.shape = <Py_ssize_t *> malloc(self.ndim * sizeof(Py_ssize_t))
        buffer.strides = <Py_ssize_t *> malloc(self.ndim * sizeof(Py_ssize_t))
        
        if not buffer.shape or not buffer.strides:
            if buffer.shape: free(buffer.shape)
            if buffer.strides: free(buffer.strides)
            raise MemoryError("Could not allocate buffer shape/strides")

        for i in range(self.ndim):
            buffer.shape[i] = self._c_tensor.shape[i]
            buffer.strides[i] = self._c_tensor.strides[i] * itemsize
            
        buffer.suboffsets = NULL
        
    def __releasebuffer__(self, Py_buffer *buffer):
        """Releases the memory allocated for the Buffer Protocol."""
        if buffer.shape != NULL:
            free(buffer.shape)
        if buffer.strides != NULL:
            free(buffer.strides)

    @property
    def data(self) -> np.ndarray:
        """Returns a zero-copy numpy view of the tensor's underlying memory."""
        return np.asarray(self)

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
        cdef size_t flat_idx = self._get_flat_index(index)
        return self._c_tensor.storage.data[flat_idx]

    def __setitem__(self, index, value):
        cdef float c_value = value
        cdef size_t flat_idx = self._get_flat_index(index)
        self._c_tensor.storage.data[flat_idx] = c_value

    # --- Python Properties ---

    @property
    def shape(self):
        if self._c_tensor == NULL: return ()
        return tuple(self._c_tensor.shape[i] for i in range(self._c_tensor.ndim))

    @property
    def strides(self):
        if self._c_tensor == NULL: return ()
        return tuple(self._c_tensor.strides[i] for i in range(self._c_tensor.ndim))

    @property
    def ndim(self):
        return self._c_tensor.ndim

    @property
    def numel(self):
        return self._c_tensor.numel

    def __repr__(self):
        return f"<cyflow.Tensor shape={self.shape} strides={self.strides} numel={self.numel} grad={self.requires_grad}>"

    # --- Math Operations & Autograd Engine ---

    @classmethod 
    def can_matmul(cls, shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]):
        """Check if two shapes can be matrix multiplied together."""
        if not shape_a or not shape_b:
            return False 
        inner_b = shape_b[-2] if len(shape_b) > 1 else shape_b[-1]
        if shape_a[-1] != inner_b:
            return False
        try:
            np.broadcast_shapes(shape_a[:-2], shape_b[:-2])
            return True
        except ValueError:
            return False

    @classmethod
    def unbroadcast(cls, grad, shape) -> np.ndarray:
        """
        Sums a gradient to match the original shape before a broadcasting operation.
        Args:
            grad: The incoming gradient (with the broadcasted shape).
            shape: The target shape (the original tensor's shape).
        Returns:
            The unbroadcasted gradient.
        """
        grad = np.asarray(grad)
        shape_tuple = tuple(shape)
        grad_shape = tuple(np.shape(grad))
        if grad_shape == shape_tuple:
            return grad

        axes = []
        ndim_diff = len(grad_shape) - len(shape_tuple)
        if ndim_diff > 0:
            axes.extend(range(0, ndim_diff))

        for i, s in enumerate(shape_tuple):
            if s == 1:
                axes.append(ndim_diff + i)

        if axes:
            grad = grad.sum(axis=tuple(axes), keepdims=True)

        try:
            return grad.reshape(shape_tuple)
        except Exception:
            raise ValueError(f"Cannot unbroadcast shape {tuple(np.shape(grad))} to {shape_tuple}")

    def __add__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_add(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Addition failed. Check if shapes are compatible.")
        
        cdef Tensor result = Tensor(shape=None, _children=(self, other), _op='+')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                np.add(self.grad, Tensor.unbroadcast(result.grad, self.shape), out=self.grad)
            if other_tensor.requires_grad:
                np.add(other_tensor.grad, Tensor.unbroadcast(result.grad, other_tensor.shape), out=other_tensor.grad)

        if result.requires_grad:
            result._backward = _backward
        return result
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_sub(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Subtraction failed. Check if shapes are compatible.")
        
        cdef Tensor result = Tensor(shape=None, _children=(self, other), _op='-')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                np.add(self.grad, Tensor.unbroadcast(result.grad, self.shape), out=self.grad)
            if other_tensor.requires_grad:
                # Note the negative sign for subtraction
                np.add(other_tensor.grad, Tensor.unbroadcast(-result.grad, other_tensor.shape), out=other_tensor.grad)

        if result.requires_grad:
            result._backward = _backward
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        cdef Tensor other_tensor = <Tensor>other
        cdef TensorImpl* result_impl = tensor_mul(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Multiplication failed. Check if shapes are compatible.")
        
        cdef Tensor result = Tensor(shape=None, _children=(self, other), _op='*')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                grad_contrib = result.grad * other_tensor.data
                np.add(self.grad, Tensor.unbroadcast(grad_contrib, self.shape), out=self.grad)
            if other_tensor.requires_grad:
                grad_contrib = result.grad * self.data
                np.add(other_tensor.grad, Tensor.unbroadcast(grad_contrib, other_tensor.shape), out=other_tensor.grad)

        if result.requires_grad:
            result._backward = _backward
        return result

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
            
        cdef TensorImpl* result_impl = tensor_pow(self._c_tensor, <int64_t>exponent)
        if result_impl == NULL:
            raise ValueError("Power operation failed.")
            
        cdef Tensor result = Tensor(shape=None, _children=(self,), _op='pow')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                # Derivative of x^n is n * x^(n-1)
                grad_contrib = result.grad * (exponent * (self.data ** (exponent - 1)))
                np.add(self.grad, Tensor.unbroadcast(grad_contrib, self.shape), out=self.grad)

        if result.requires_grad:
            result._backward = _backward
        return result
    
    def exp(self):
        cdef TensorImpl* result_impl = tensor_exp(self._c_tensor)
        if result_impl == NULL:
            raise ValueError("Exponential operation failed.")
            
        cdef Tensor result = Tensor(shape=None, _children=(self,), _op='exp')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                # Derivative of exp(x) is exp(x), which is stored in result.data
                grad_contrib = result.grad * result.data
                np.add(self.grad, Tensor.unbroadcast(grad_contrib, self.shape), out=self.grad)

        if result.requires_grad:
            result._backward = _backward
        return result
    

        if result.requires_grad:
            result._backward = _backward
        return result

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
            # Compute the sum using the zero-copy numpy view
            out_data = np.sum(self.data, axis=axis, keepdims=keepdims, dtype=np.float32)
            out_data = np.atleast_1d(out_data)
            # Pass the numpy array to the first argument (which acts as shape/data)
            out = Tensor(out_data, _children=(self,), _op='sum')
            
            # Ensure the gradient is initialized if needed
            if out.requires_grad:
                out.grad = np.zeros(out.shape, dtype=np.float32)

            def _backward():
                if self.requires_grad:
                    if axis is None:  # scalar result
                        grad_to_expand = out.grad
                    else:
                        grad_to_expand = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
                    
                    # Add into self.grad with broadcasting to avoid temporaries
                    np.add(self.grad, grad_to_expand, out=self.grad)
            
            if out.requires_grad:
                out._backward = _backward
                
            return out
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented

        cdef Tensor other_tensor = <Tensor>other
        if not Tensor.can_matmul(self.shape, other_tensor.shape):
            raise ValueError(f"Shapes {self.shape} and {other_tensor.shape} not aligned for matmul")

        cdef TensorImpl* result_impl = tensor_matmul(self._c_tensor, other_tensor._c_tensor)
        if result_impl == NULL:
            raise ValueError("Matrix multiplication failed. Check if shapes are compatible.")

        cdef Tensor result = Tensor(shape=None, _children=(self, other), _op='@')
        result._c_tensor = result_impl
        if result.requires_grad:
            result.grad = np.zeros(result.shape, dtype=np.float32)

        def _backward():
            if self.requires_grad:
                if other_tensor.ndim > 1:
                    other_transposed = np.swapaxes(other_tensor.data, -1, -2)
                    self_grad_contrib = result.grad @ other_transposed
                else:
                    if result.grad.ndim == 0:
                        self_grad_contrib = result.grad * other_tensor.data
                    else:
                        self_grad_contrib = np.outer(result.grad, other_tensor.data)
                
                np.add(self.grad, Tensor.unbroadcast(self_grad_contrib, self.shape), out=self.grad)

            if other_tensor.requires_grad:
                if self.ndim > 1:
                    self_transposed = np.swapaxes(self.data, -1, -2)
                    other_grad_contrib = self_transposed @ result.grad
                else:
                    if result.grad.ndim == 0:
                        other_grad_contrib = result.grad * self.data
                    else:
                        other_grad_contrib = np.outer(self.data, result.grad)

                np.add(other_tensor.grad, Tensor.unbroadcast(other_grad_contrib, other_tensor.shape), out=other_tensor.grad)

        if result.requires_grad:
            result._backward = _backward
        return result

    def backward(self) -> None:
        """
        Performs backpropagation starting from this tensor.
        Assumes this tensor is the final output (e.g., a scalar loss).
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor that does not require_grad")
            
        # Build topological sort
        topo = []
        visited = set()

        def build_topo(v: 'Tensor'):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        # --- Initialize Gradients ---
        for node in topo:
            is_leaf = len(node._prev) == 0
            
            if not is_leaf:
                # Intermediate nodes MUST be zeroed every backward pass 
                # to prevent incorrect double-counting in the chain rule.
                node.grad = np.zeros_like(node.data)
            else:
                # Leaf nodes (weights/biases) ACCUMULATE. 
                # We just ensure the array exists, but do NOT zero it.
                if node.grad is None:
                    node.grad = np.zeros_like(node.data)

        # Set the seed gradient for the output tensor
        if len(self._prev) == 0:
            # Edge case: If the loss itself is a leaf node, accumulate the seed
            np.add(self.grad, np.ones_like(self.data), out=self.grad)
        else:
            # Standard case: The loss is an intermediate node. Set it to 1s.
            self.grad = np.ones_like(self.data)

        # --- Propagate Gradients ---
        for node in reversed(topo):
            node._backward()