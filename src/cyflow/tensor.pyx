# cython: language_level=3
from typing import Tuple, Set, Optional,Union
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
                # Ensure float32
                if not np.issubdtype(shape.dtype, np.floating) or shape.dtype != np.float32:
                    shape = np.asarray(shape, dtype=np.float32)
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
        cdef TensorImpl* result_impl = NULL
        other_tensor: Tensor = other
        result_impl = tensor_add(self._c_tensor, (<Tensor>other_tensor)._c_tensor)
        if result_impl == NULL:
            raise ValueError("Addition failed. Check if shapes are compatible.")

        result = Tensor(shape=None, _children=(self, other), _op='+')
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
        cdef TensorImpl* result_impl = NULL
        other_tensor: Tensor = other
        result_impl = tensor_sub(self._c_tensor, (<Tensor>other_tensor)._c_tensor)
        if result_impl == NULL:
            raise ValueError("Subtraction failed. Check if shapes are compatible.")

        result = Tensor(shape=None, _children=(self, other), _op='-')
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

    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise multiplication with scalar or tensor support."""
        cdef TensorImpl* result_impl = NULL
        # Tensor * Tensor
        if isinstance(other, Tensor):
            other_tensor: Tensor = other
            result_impl = tensor_mul(self._c_tensor, (<Tensor>other_tensor)._c_tensor)
            if result_impl == NULL:
                raise ValueError("Multiplication failed. Check shapes.")

            result = Tensor(shape=None, _children=(self, other), _op='*',
                           requires_grad=self.requires_grad or other_tensor.requires_grad)
            result._c_tensor = result_impl

            if result.requires_grad:
                def _backward():
                    if self.requires_grad:
                        grad_contrib = result.grad * other_tensor.data
                        np.add(self.grad, Tensor.unbroadcast(grad_contrib, self.shape), out=self.grad)
                    if other_tensor.requires_grad:
                        grad_contrib = result.grad * self.data
                        np.add(other_tensor.grad, Tensor.unbroadcast(grad_contrib, other_tensor.shape), out=other_tensor.grad)
                result._backward = _backward
            return result

        # Tensor * scalar (int/float) or ndarray
        if isinstance(other, (int, float, np.ndarray)):
            scalar = float(other) if isinstance(other, (int, float)) else other
            result = Tensor(self.data * scalar, _children=(self,), _op='*',
                           requires_grad=self.requires_grad)
            if result.requires_grad:
                def _backward():
                    if self.requires_grad:
                        grad_contrib = scalar * result.grad
                        np.add(self.grad, Tensor.unbroadcast(grad_contrib, self.shape), out=self.grad)
                result._backward = _backward
            return result

        return NotImplemented
    def __rmul__(self, other):
        return self.__mul__(other)
    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")

        cdef TensorImpl* result_impl = tensor_pow(self._c_tensor, <int64_t>exponent)
        if result_impl == NULL:
            raise ValueError("Power operation failed.")

        result = Tensor(shape=None, _children=(self,), _op='pow')
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

        result = Tensor(shape=None, _children=(self,), _op='exp')
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

        cdef TensorImpl* result_impl = NULL
        other_tensor: Tensor = other
        if not Tensor.can_matmul(self.shape, other_tensor.shape):
            raise ValueError(f"Shapes {self.shape} and {other_tensor.shape} not aligned for matmul")

        result_impl = tensor_matmul(self._c_tensor, (<Tensor>other_tensor)._c_tensor)
        if result_impl == NULL:
            raise ValueError("Matrix multiplication failed. Check if shapes are compatible.")

        result = Tensor(shape=None, _children=(self, other), _op='@')
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

    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        """Element-wise division (``self/other``)"""
        if not isinstance(other, Tensor):
            # Handle scalar division
            out = Tensor(self.data / other, _children = (self,),_op ='/')
            if out.requires_grad:
                def _backward_scalar():
                    if self.requires_grad:
                        np.add(self.grad, Tensor.unbroadcast((1.0 / other) * out.grad, self.shape), out=self.grad)
                out._backward = _backward_scalar
            return out

        # Handle Tensor division
        out = Tensor(self.data / other.data, _children = (self, other), _op = '/')
        if out.requires_grad:
            def _backward_tensor():
                if self.requires_grad:
                    np.add(self.grad, Tensor.unbroadcast((1.0 / other.data) * out.grad, self.shape), out=self.grad)
                if other.requires_grad:
                    np.add(other.grad, Tensor.unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.shape), out=other.grad)
            out._backward = _backward_tensor
        return out

    def log(self) -> 'Tensor':
        """Natural logarithm (ln)."""
        out = Tensor(np.log(self.data), (self,), 'log')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, (1.0 / (self.data + 1e-8)) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def log10(self) -> 'Tensor':
        """Base-10 logarithm."""
        out = Tensor(np.log10(self.data), (self,), 'log10')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, (1.0 / ((self.data + 1e-8) * np.log(10.0))) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def sqrt(self) -> 'Tensor':
        """Square root."""
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, (0.5 / (np.sqrt(self.data) + 1e-8)) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """Clip tensor values to [min_val, max_val]."""
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), 'clip')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    mask = (self.data >= min_val) & (self.data <= max_val)
                    np.add(self.grad, out.grad * mask, out=self.grad)
            out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        """Rectified Linear Unit: max(0, x)."""
        out = Tensor(np.maximum(self.data, 0.0), _children =(self,), _op = 'relu')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, (self.data > 0).astype(np.float32) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        """Leaky ReLU: x if x > 0, else alpha * x."""
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), _children = (self,), _op ='leaky_relu')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, np.where(self.data > 0, 1.0, alpha) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def elu(self, alpha: float = 1.0) -> 'Tensor':
        """Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1)."""
        out = Tensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), _children = (self,), _op = 'elu')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, np.where(self.data > 0, 1.0, alpha * np.exp(self.data)) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def selu(self, alpha: float = 1.67326, scale: float = 1.0507) -> 'Tensor':
        """Scaled Exponential Linear Unit."""
        out = Tensor(scale * np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), _children =(self,), _op ='selu')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, scale * np.where(self.data > 0, 1.0, alpha * np.exp(self.data)) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation: 1 / (1 + exp(-x))."""
        # Numerically stable sigmoid
        sig = np.where(self.data >= 0,
                       1.0 / (1.0 + np.exp(-self.data)),
                       np.exp(self.data) / (1.0 + np.exp(self.data)))
        out = Tensor(sig, _children =(self,), _op = 'sigmoid')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, sig * (1.0 - sig) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        """Hyperbolic tangent activation."""
        t = np.tanh(self.data)
        out = Tensor(t, _children = (self,), _op ='tanh')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, (1.0 - t ** 2) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def swish(self) -> 'Tensor':
        """Swish activation: x * sigmoid(x)."""
        sig = self.sigmoid()
        out = self * sig
        out._op = 'swish'
        return out

    def gelu(self) -> 'Tensor':
        """Gaussian Error Linear Unit."""
        from scipy import special as sp
        data_np = np.asarray(self.data)
        erf_result = sp.erf(data_np / np.sqrt(2.0))
        out_data = 0.5 * self.data * (1.0 + erf_result)
        out = Tensor(out_data, _children =(self,), _op ='gelu')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    sqrt_2pi = np.sqrt(2.0 * np.pi)
                    cdf = 0.5 * (1.0 + sp.erf(data_np / np.sqrt(2.0)))
                    pdf = (1.0 / sqrt_2pi) * np.exp(-0.5 * data_np ** 2)
                    np.add(self.grad, (cdf + data_np * pdf) * out.grad, out=self.grad)
            out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax: exp(x_i) / sum(exp(x_j)) along axis."""
        max_val = self.data.max(axis=axis, keepdims=True)
        e_x = np.exp(self.data - max_val)
        sum_e_x = e_x.sum(axis=axis, keepdims=True)
        sm = e_x / (sum_e_x + 1e-8)
        out = Tensor(sm, _children = (self,), _op = 'softmax')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    y = out.data
                    g = out.grad
                    sum_gy = (g * y).sum(axis=axis, keepdims=True)
                    grad_contrib = y * (g - sum_gy)
                    np.add(self.grad, grad_contrib, out=self.grad)
            out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """Log-softmax: numerically stable log(softmax(x))."""
        max_val = self.data.max(axis=axis, keepdims=True)
        x_minus_max = self.data - max_val
        log_sum_exp = np.log(np.exp(x_minus_max).sum(axis=axis, keepdims=True) + 1e-8)
        log_sm = x_minus_max - log_sum_exp
        out = Tensor(log_sm, _children = (self,), _op = 'log_softmax')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    g = out.grad
                    sm = np.exp(out.data)
                    grad_contrib = g - sm * g.sum(axis=axis, keepdims=True)
                    np.add(self.grad, grad_contrib, out=self.grad)
            out._backward = _backward
        return out

    def reshape(self, *new_shape: int) -> 'Tensor':
        """Reshape tensor to new shape."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])

        if -1 in new_shape:
            new_shape_list = list(new_shape)
            known_prod = np.prod([d for d in new_shape_list if d != -1])
            new_shape_list[new_shape_list.index(-1)] = self.data.size // int(known_prod)
            new_shape = tuple(new_shape_list)

        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    np.add(self.grad, out.grad.reshape(self.shape), out=self.grad)
            out._backward = _backward
        return out

    def view(self, *new_shape: int) -> 'Tensor':
        """View tensor with new shape (wrapper for reshape)."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        return self.reshape(*new_shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Permute tensor dimensions."""
        out = Tensor(np.transpose(self.data, axes=axes), _children = (self,), _op = 'transpose')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if axes is None:
                        inverse_axes = None
                    else:
                        inverse_axes = tuple(np.argsort(axes))
                    np.add(self.grad, np.transpose(out.grad, axes=inverse_axes), out=self.grad)
            out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """Compute mean along axis."""
        if axis is None:
            n = float(self.numel)
        elif isinstance(axis, int):
            n = float(self.shape[axis])
        else:
            n = float(np.prod([self.shape[i] for i in axis]))

        sum_out = self.sum(axis=axis, keepdims=keepdims)
        out = sum_out * (1.0 / n)          # Now works reliably
        out._op = 'mean'
        return out

    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample variance (N-1 denominator)."""
        mean = self.mean(axis=axis, keepdims=True)
        diff = self - mean
        sq_diff = diff ** 2

        if axis is None:
            n = self.numel
        elif isinstance(axis, int):
            n = self.shape[axis]
        else:
            n = int(np.prod([self.shape[a] for a in axis]))

        denom = max(n - 1, 1)
        var = sq_diff.sum(axis=axis, keepdims=keepdims) / float(denom)
        var._op = 'var'
        return var

    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = True) -> 'Tensor':
        """Sample standard deviation."""
        variance = self.var(axis=axis, keepdims=keepdims)
        return variance.sqrt()

    def item(self) -> float:
        """Return scalar value (only for single-element tensors)."""
        if self.data.size != 1:
            raise ValueError("item() can only be called on tensors with one element.")
        return float(self.data.flat[0])

    def __getitem__(self, indices) -> 'Tensor':
        """Get item by index/slice."""
        out = Tensor(self.data[indices], (self,), 'getitem')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_slice = np.zeros_like(self.data)
                    grad_slice[indices] = out.grad
                    np.add(self.grad, grad_slice, out=self.grad)
            out._backward = _backward
        return out

    def bool(self) -> 'Tensor':
        """Cast to boolean (detached)."""
        return Tensor(self.data.astype(bool), requires_grad=False)

    def __bool__(self) -> bool:
        """Boolean context behavior."""
        if self.data.size == 1:
            return bool(self.data.item())
        raise ValueError(
            "The truth value of a Tensor with more than one element is ambiguous. "
            "Use .any() or .all() if you want to check for element-wise truth."
        )

    def masked_fill(self, mask: 'Tensor', fill_value: float) -> 'Tensor':
        """Fill elements where mask is True with fill_value."""
        out = Tensor(np.where(mask.data, fill_value, self.data), _children = (self,), _op = 'masked_fill')
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_for_self = np.where(mask.data, 0.0, out.grad)
                    np.add(self.grad, grad_for_self, out=self.grad)
            out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        return self * -1
    def numpy(self) -> np.ndarray:
        """Returns a copy of the tensor's data as a NumPy array."""
        if self._c_tensor == NULL:
            return np.empty((0,), dtype=np.float32)
        # Fast path: leverage zero-copy buffer view and make a copy
        return np.array(self.data, copy=True)

    def __repr__(self) -> str:
        """Returns a human-readable string representation of the Tensor."""
        if self._c_tensor == NULL:
            return "Tensor(uninitialized)"

        # Use self.data (zero-copy view) for numpy string formatting
        data_np = self.data
        data_str = np.array2string(data_np, max_line_width=70, precision=4, suppress_small=True)

        # Format multi-line array representations cleanly
        if '\n' in data_str:
            lines = data_str.split('\n')
            data_str = f"{lines[0]} ... {lines[-1].strip()}"

        grad_info = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor(data={data_str}, shape={self.shape}, requires_grad={self.requires_grad}{grad_info})"

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
