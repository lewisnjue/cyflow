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

        # 3. Expand Ellipsis (...) into explicit slice(None) objects
        clean_key = []
        missing_axes = self.ndim - explicit_axes

        for item in tuple_key:
            if item is Ellipsis:
                for _ in range(missing_axes):
                    clean_key.append(slice(None))
            else:
                clean_key.append(item)

        # 4. Calculate output dimensions and allocate shape/strides memory
        out_ndim = (self.ndim - num_int) + num_none

        if out_ndim > 0:
            c_shape = <int64_t*>malloc(out_ndim * sizeof(int64_t))
            c_strides = <int64_t*>malloc(out_ndim * sizeof(int64_t))
            if not c_shape or not c_strides:
                if c_shape: free(c_shape)
                if c_strides: free(c_strides)
                raise MemoryError("Failed to allocate shape/stride memory for view")

        # 5. Single-pass loop calculating shapes, strides, and memory offsets
        for item in clean_key:
            if item is None:
                # Insert a new dimension of size 1 (np.newaxis / unsqueeze)
                c_shape[dst_dim] = 1
                if src_dim < self.ndim:
                    c_strides[dst_dim] = self._tensor.strides[src_dim]
                else:
                    c_strides[dst_dim] = 1
                dst_dim += 1
                # src_dim is NOT incremented (doesn't consume source dimension)

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
                # dst_dim is NOT incremented (integer indexing collapses the dimension)

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
