# cython: language_level=3
from libc.stdlib cimport malloc, free, calloc
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
        size_t storage_offset

    void compute_contiguous_strides(int64_t *strides, const int64_t *shape, size_t ndim)
    TensorImpl *tensor_view(TensorImpl *src, const int64_t *shape, size_t ndim)
    TensorImpl *tensor_index(TensorImpl *src, int64_t index)


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

cdef extern from "cuda_runtime.h":
    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
    int cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)

CPU = DEVICE_CPU
CUDA = DEVICE_CUDA

cpdef manual_seed(unsigned long long seed, int device=CPU):
    if device == DEVICE_CPU:
        cyflow_manual_seed(seed)
    elif device == DEVICE_CUDA:
        cyflow_manual_seed_cuda(seed)
    else:
        raise ValueError(f"Unsupported device integer: {device}")

def _get_nested_list_shape_and_flat(lst):
    if not isinstance(lst, (list, tuple)):
        raise TypeError("Expected list or tuple")

    shape = []
    curr = lst
    while isinstance(curr, (list, tuple)):
        shape.append(len(curr))
        if len(curr) == 0:
            break
        curr = curr[0]

    flat = []
    def _flatten(item, depth=0):
        if isinstance(item, (list, tuple)):
            if depth < len(shape) and len(item) != shape[depth]:
                raise ValueError(
                    f"Inconsistent list dimension at depth {depth}: expected {shape[depth]}, got {len(item)}"
                )
            for sub in item:
                _flatten(sub, depth + 1)
        elif isinstance(item, (int, float)):
            flat.append(float(item))
        else:
            raise TypeError(f"Invalid element type in list: {type(item).__name__}")

    _flatten(lst)
    return tuple(shape), flat

cdef class Tensor:
    cdef TensorImpl* _tensor

    def __cinit__(self, shape=None, int device=CPU):
        cdef size_t ndim
        cdef int64_t* c_shape
        cdef int i

        if shape is None:
            self._tensor = NULL
            return

        ndim = len(shape)
        c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
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

    def item(self):
        cdef float val = 0.0
        cdef float* data_ptr = NULL

        if self._tensor is NULL:
            raise ValueError("Cannot call item() on an uninitialized tensor")

        if self._tensor.numel != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars (got numel {self._tensor.numel})"
            )

        data_ptr = <float*>self._tensor.storage.data

        if self._tensor.storage.device == DEVICE_CPU:
            val = data_ptr[self._tensor.storage_offset]
        elif self._tensor.storage.device == DEVICE_CUDA:
            cudaMemcpy(
                &val,
                data_ptr + self._tensor.storage_offset,
                sizeof(float),
                cudaMemcpyDeviceToHost
            )

        return float(val)
    
    cpdef _to_nested_list(self):
            # Use Python properties (self.ndim, self.shape) instead of raw C pointers
            if self.ndim == 0:
                return self.item()
            elif self.ndim == 1:
                return [self[i].item() for i in range(self.shape[0])]
            else:
                return [self[i]._to_nested_list() for i in range(self.shape[0])]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self._tensor is NULL:
            return "<Tensor [Uninitialized]>"
        
        # If the tensor is small enough, display its data values nicely
        if self.numel <= 100:
            try:
                data = self._to_nested_list()
                return f"<Tensor data={data}, shape={self.shape}, device='{self.device}'>"
            except Exception as e:
                pass

        # Default summary representation for large tensors
        return f"<Tensor shape={self.shape}, strides={self.strides}, device='{self.device}'>"
    @property
    def nbytes(self):
        return self.numel * sizeof(float)

    def fill_uniform(self):
        if self._tensor.storage.device == DEVICE_CPU:
            tensor_fill_uniform_cpu(self._tensor)
        elif self._tensor.storage.device == DEVICE_CUDA:
            tensor_fill_uniform_cuda(self._tensor)
    cdef _fill_scalar(self, float val):
        cdef size_t numel = self._tensor.numel
        cdef size_t ndim = self._tensor.ndim
        cdef int64_t* shape = self._tensor.shape
        cdef int64_t* strides = self._tensor.strides
        cdef int64_t offset = self._tensor.storage_offset
        cdef int device = self._tensor.storage.device

        cdef float* target_ptr = NULL
        cdef int64_t* indices = NULL
        cdef size_t elem_i, k
        cdef int64_t cur_offset

        if numel == 0:
            return

        if ndim > 0:
            indices = <int64_t*>calloc(ndim, sizeof(int64_t))
            if not indices:
                raise MemoryError("Failed to allocate index buffer")

        try:
            target_ptr = <float*>self._tensor.storage.data
            for elem_i in range(numel):
                cur_offset = offset
                for k in range(ndim):
                    cur_offset += indices[k] * strides[k]

                if device == DEVICE_CPU:
                    target_ptr[cur_offset] = val
                elif device == DEVICE_CUDA:
                    cudaMemcpy(
                        target_ptr + cur_offset,
                        &val,
                        sizeof(float),
                        cudaMemcpyHostToDevice
                    )

                if ndim > 0:
                    for k in range(ndim - 1, -1, -1):
                        indices[k] += 1
                        if indices[k] < shape[k]:
                            break
                        indices[k] = 0
        finally:
            if indices:
                free(indices)

    cdef _fill_from_flat_list(self, list flat_vals):
        cdef size_t numel = self._tensor.numel
        cdef size_t ndim = self._tensor.ndim
        cdef int64_t* shape = self._tensor.shape
        cdef int64_t* strides = self._tensor.strides
        cdef int64_t offset = self._tensor.storage_offset
        cdef int device = self._tensor.storage.device

        cdef float* target_ptr = NULL
        cdef int64_t* indices = NULL
        cdef size_t elem_i, k
        cdef int64_t cur_offset
        cdef float val

        if numel != len(flat_vals):
            raise ValueError(f"Expected {numel} elements, got {len(flat_vals)}")

        if numel == 0:
            return

        if ndim > 0:
            indices = <int64_t*>calloc(ndim, sizeof(int64_t))
            if not indices:
                raise MemoryError("Failed to allocate index buffer")

        try:
            target_ptr = <float*>self._tensor.storage.data
            for elem_i in range(numel):
                val = float(flat_vals[elem_i])
                cur_offset = offset
                for k in range(ndim):
                    cur_offset += indices[k] * strides[k]

                if device == DEVICE_CPU:
                    target_ptr[cur_offset] = val
                elif device == DEVICE_CUDA:
                    cudaMemcpy(
                        target_ptr + cur_offset,
                        &val,
                        sizeof(float),
                        cudaMemcpyHostToDevice
                    )

                if ndim > 0:
                    for k in range(ndim - 1, -1, -1):
                        indices[k] += 1
                        if indices[k] < shape[k]:
                            break
                        indices[k] = 0
        finally:
            if indices:
                free(indices)

    cdef _copy_from_tensor(self, Tensor src):
        cdef size_t numel = self._tensor.numel
        cdef size_t ndim = self._tensor.ndim
        cdef int64_t* shape = self._tensor.shape
        cdef int64_t* strides = self._tensor.strides
        cdef int64_t offset = self._tensor.storage_offset
        cdef int device = self._tensor.storage.device

        cdef size_t src_ndim = src._tensor.ndim
        cdef int64_t* src_strides = src._tensor.strides
        cdef int64_t src_offset = src._tensor.storage_offset
        cdef int src_device = src._tensor.storage.device

        cdef float* target_ptr = <float*>self._tensor.storage.data
        cdef float* src_ptr = <float*>src._tensor.storage.data

        cdef int64_t* target_indices = NULL
        cdef int64_t* src_indices = NULL
        cdef size_t elem_i, k
        cdef int64_t cur_target_offset, cur_src_offset

        if self.shape != src.shape:
            raise ValueError(f"Cannot copy tensor of shape {src.shape} to tensor of shape {self.shape}")

        if device != src_device:
            raise ValueError(f"Cannot copy between different devices: {self.device} vs {src.device}")

        if numel == 0:
            return

        if ndim > 0:
            target_indices = <int64_t*>calloc(ndim, sizeof(int64_t))
            src_indices = <int64_t*>calloc(src_ndim, sizeof(int64_t))
            if not target_indices or not src_indices:
                if target_indices: free(target_indices)
                if src_indices: free(src_indices)
                raise MemoryError("Failed to allocate index buffer")

        try:
            for elem_i in range(numel):
                cur_target_offset = offset
                for k in range(ndim):
                    cur_target_offset += target_indices[k] * strides[k]

                cur_src_offset = src_offset
                for k in range(src_ndim):
                    cur_src_offset += src_indices[k] * src_strides[k]

                if device == DEVICE_CPU:
                    target_ptr[cur_target_offset] = src_ptr[cur_src_offset]
                elif device == DEVICE_CUDA:
                    cudaMemcpy(
                        target_ptr + cur_target_offset,
                        src_ptr + cur_src_offset,
                        sizeof(float),
                        cudaMemcpyDeviceToDevice
                    )

                if ndim > 0:
                    for k in range(ndim - 1, -1, -1):
                        target_indices[k] += 1
                        if target_indices[k] < shape[k]:
                            break
                        target_indices[k] = 0

                if src_ndim > 0:
                    for k in range(src_ndim - 1, -1, -1):
                        src_indices[k] += 1
                        if src_indices[k] < src._tensor.shape[k]:
                            break
                        src_indices[k] = 0
        finally:
            if target_indices: free(target_indices)
            if src_indices: free(src_indices)

    cpdef _apply_inplace(self, object other, str op):
        cdef size_t numel = self._tensor.numel
        cdef size_t ndim = self._tensor.ndim
        cdef int64_t* shape = self._tensor.shape
        cdef int64_t* strides = self._tensor.strides
        cdef int64_t offset = self._tensor.storage_offset
        cdef int device = self._tensor.storage.device
        
        cdef float* target_ptr = <float*>self._tensor.storage.data
        cdef int64_t* indices = NULL
        cdef size_t elem_i, k
        cdef int64_t cur_offset, cur_src_offset
        cdef float val = 0.0
        cdef float src_val = 0.0

        cdef Tensor src_tensor = None
        cdef size_t src_ndim = 0
        cdef int64_t* src_strides = NULL
        cdef int64_t src_offset = 0
        cdef float* src_ptr = NULL
        cdef int64_t* src_indices = NULL

        # 1. Parse operand (Scalar vs Tensor)
        if isinstance(other, Tensor):
            src_tensor = <Tensor>other
            if self.shape != src_tensor.shape:
                raise ValueError(f"Shape mismatch for in-place operation: {self.shape} vs {src_tensor.shape}")
            if self.device != src_tensor.device:
                raise ValueError(f"Device mismatch for in-place operation: {self.device} vs {src_tensor.device}")
            
            src_ndim = src_tensor._tensor.ndim
            src_strides = src_tensor._tensor.strides
            src_offset = src_tensor._tensor.storage_offset
            src_ptr = <float*>src_tensor._tensor.storage.data
            if src_ndim > 0:
                src_indices = <int64_t*>calloc(src_ndim, sizeof(int64_t))
        elif isinstance(other, (int, float)):
            src_val = float(other)
        else:
            raise TypeError(f"Unsupported operand type for in-place operation: {type(other).__name__}")

        if numel == 0:
            return self

        if ndim > 0:
            indices = <int64_t*>calloc(ndim, sizeof(int64_t))
            if not indices:
                if src_indices: free(src_indices)
                raise MemoryError("Failed to allocate index buffer")

        try:
            for elem_i in range(numel):
                cur_offset = offset
                for k in range(ndim):
                    cur_offset += indices[k] * strides[k]

                # Fetch source value if operand is a Tensor
                if src_tensor is not None:
                    cur_src_offset = src_offset
                    for k in range(src_ndim):
                        cur_src_offset += src_indices[k] * src_strides[k]
                    
                    if device == DEVICE_CPU:
                        src_val = src_ptr[cur_src_offset]
                    elif device == DEVICE_CUDA:
                        cudaMemcpy(&src_val, src_ptr + cur_src_offset, sizeof(float), cudaMemcpyDeviceToHost)

                # Fetch current target value
                if device == DEVICE_CPU:
                    val = target_ptr[cur_offset]
                elif device == DEVICE_CUDA:
                    cudaMemcpy(&val, target_ptr + cur_offset, sizeof(float), cudaMemcpyDeviceToHost)

                # Perform arithmetic operation
                if op == "+":
                    val += src_val
                elif op == "-":
                    val -= src_val
                elif op == "*":
                    val *= src_val
                elif op == "/":
                    if src_val == 0.0:
                        raise ZeroDivisionError("Division by zero in tensor in-place division")
                    val /= src_val

                # Write result back to target
                if device == DEVICE_CPU:
                    target_ptr[cur_offset] = val
                elif device == DEVICE_CUDA:
                    cudaMemcpy(target_ptr + cur_offset, &val, sizeof(float), cudaMemcpyHostToDevice)

                # Advance indices for multi-dimensional traversal
                if ndim > 0:
                    for k in range(ndim - 1, -1, -1):
                        indices[k] += 1
                        if indices[k] < shape[k]:
                            break
                        indices[k] = 0

                if src_indices != NULL:
                    for k in range(src_ndim - 1, -1, -1):
                        src_indices[k] += 1
                        if src_indices[k] < src_tensor._tensor.shape[k]:
                            break
                        src_indices[k] = 0
        finally:
            if indices: free(indices)
            if src_indices: free(src_indices)

        return self

    def __iadd__(self, other):
        return self._apply_inplace(other, "+")

    def __isub__(self, other):
        return self._apply_inplace(other, "-")

    def __imul__(self, other):
        return self._apply_inplace(other, "*")

    def __itruediv__(self, other):
        return self._apply_inplace(other, "/")

    def __setitem__(self, key, value):
        # 1. Resolve sub-view for the given key
        cdef Tensor target = self[key]

        # 2. Assign scalar float/int
        if isinstance(value, (int, float)):
            target._fill_scalar(float(value))

        # 3. Assign Tensor
        elif isinstance(value, Tensor):
            if target.shape != (<Tensor>value).shape:
                raise ValueError(
                    f"Shape mismatch: cannot assign Tensor with shape {(<Tensor>value).shape} to target view with shape {target.shape}"
                )
            target._copy_from_tensor(<Tensor>value)

        # 4. Assign Python list / tuple
        elif isinstance(value, (list, tuple)):
            list_shape, flat_vals = _get_nested_list_shape_and_flat(value)

            if target.shape != list_shape:
                raise ValueError(
                    f"Shape mismatch: cannot assign list with shape {list_shape} to target view with shape {target.shape}"
                )

            target._fill_from_flat_list(flat_vals)

        else:
            raise TypeError(f"Cannot assign value of type {type(value).__name__} to Tensor")

    def view(self, *shape) -> Tensor:
        cdef size_t target_numel = 1
        cdef size_t ndim
        cdef int64_t* c_shape
        cdef TensorImpl* new_impl = NULL
        cdef int i

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        for dim in target_shape:
            target_numel *= dim

        if target_numel != self.numel:
            raise ValueError(f"Cannot reshape tensor of size {self.numel} into shape {target_shape}")

        ndim = len(target_shape)
        c_shape = <int64_t*>malloc(ndim * sizeof(int64_t))
        if not c_shape:
            raise MemoryError("Failed to allocate memory for shape array")

        for i in range(ndim):
            c_shape[i] = target_shape[i]

        try:
            new_impl = tensor_view(self._tensor, c_shape, ndim)
        finally:
            free(c_shape)

        if new_impl is NULL:
            raise RuntimeError("Backend failed to create tensor view")

        return Tensor._from_c_tensor(new_impl)

    def _set_data_from_list(self, flat_data: list):
        cdef size_t numel
        cdef float* c_data
        cdef int i

        if len(flat_data) != self.numel:
            raise ValueError(f"Expected {self.numel} elements, got {len(flat_data)}")

        numel = self.numel
        c_data = <float*>malloc(numel * sizeof(float))
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

    def __getitem__(self, key):
        cdef tuple tuple_key
        cdef list clean_key
        cdef int num_none = 0
        cdef int num_int = 0
        cdef int num_slice = 0
        cdef int explicit_axes = 0
        cdef bint has_ellipsis = False
        cdef int missing_axes = 0
        cdef size_t out_ndim = 0
        cdef size_t src_dim = 0
        cdef size_t dst_dim = 0
        cdef int64_t offset_delta = 0
        cdef size_t out_numel = 1
        cdef Py_ssize_t start, stop, step, length, idx
        cdef int64_t cur_dim_size, cur_stride
        cdef int64_t* c_shape = NULL
        cdef int64_t* c_strides = NULL
        cdef TensorImpl* result = NULL

        if not isinstance(key, tuple):
            tuple_key = (key,)
        else:
            tuple_key = key

        for item in tuple_key:
            if isinstance(item, (list, Tensor)):
                raise NotImplementedError("Advanced indexing (lists or Tensors) is not supported")
            elif item is Ellipsis:
                if has_ellipsis:
                    raise IndexError("An index can only have a single ellipsis ('...')")
                has_ellipsis = True
            elif item is None:
                num_none += 1
            elif isinstance(item, int):
                num_int += 1
                explicit_axes += 1
            elif isinstance(item, slice):
                num_slice += 1
                explicit_axes += 1
            else:
                raise TypeError(f"Invalid index type: {type(item).__name__}")

        if explicit_axes > self.ndim:
            raise IndexError(f"Too many indices for tensor: tensor is {self.ndim}D, but {explicit_axes} axes were indexed")

        clean_key = []
        missing_axes = self.ndim - explicit_axes

        for item in tuple_key:
            if item is Ellipsis:
                for _ in range(missing_axes):
                    clean_key.append(slice(None))
            else:
                clean_key.append(item)

        out_ndim = (self.ndim - num_int) + num_none

        if out_ndim > 0:
            c_shape = <int64_t*>malloc(out_ndim * sizeof(int64_t))
            c_strides = <int64_t*>malloc(out_ndim * sizeof(int64_t))
            if not c_shape or not c_strides:
                if c_shape: free(c_shape)
                if c_strides: free(c_strides)
                raise MemoryError("Failed to allocate shape/stride memory for view")

        for item in clean_key:
            if item is None:
                c_shape[dst_dim] = 1
                if src_dim < self.ndim:
                    c_strides[dst_dim] = self._tensor.strides[src_dim]
                else:
                    c_strides[dst_dim] = 1
                dst_dim += 1

            elif isinstance(item, int):
                cur_dim_size = self._tensor.shape[src_dim]
                cur_stride = self._tensor.strides[src_dim]
                idx = item
                if idx < 0:
                    idx += cur_dim_size
                if idx < 0 or idx >= cur_dim_size:
                    if c_shape: free(c_shape)
                    if c_strides: free(c_strides)
                    raise IndexError(f"Index {item} is out of bounds for axis {src_dim} with size {cur_dim_size}")

                offset_delta += idx * cur_stride
                src_dim += 1

            elif isinstance(item, slice):
                cur_dim_size = self._tensor.shape[src_dim]
                cur_stride = self._tensor.strides[src_dim]

                start, stop, step = item.indices(cur_dim_size)
                if step > 0:
                    length = (stop - start + step - 1) // step if stop > start else 0
                else:
                    length = (start - stop + (-step) - 1) // (-step) if stop < start else 0

                c_shape[dst_dim] = length
                c_strides[dst_dim] = cur_stride * step
                out_numel *= <size_t>length
                offset_delta += start * cur_stride

                dst_dim += 1
                src_dim += 1

        while src_dim < self.ndim:
            c_shape[dst_dim] = self._tensor.shape[src_dim]
            c_strides[dst_dim] = self._tensor.strides[src_dim]
            out_numel *= <size_t>self._tensor.shape[src_dim]
            dst_dim += 1
            src_dim += 1

        result = <TensorImpl*>malloc(sizeof(TensorImpl))
        if not result:
            if c_shape: free(c_shape)
            if c_strides: free(c_strides)
            raise MemoryError("Failed to allocate TensorImpl for view")

        result.storage = self._tensor.storage
        result.storage.ref_count += 1
        result.storage_offset = self._tensor.storage_offset + offset_delta
        result.ndim = out_ndim
        result.numel = out_numel
        result.shape = c_shape
        result.strides = c_strides

        return Tensor._from_c_tensor(result)
