#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuRAND Error: %d at %s:%d\n", (int)err_, __FILE__,      \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#ifndef MAX_DIMS
#define MAX_DIMS 8
#endif

#define CUDA_THREADS_PER_BLOCK 256

static curandGenerator_t cuda_gen;
static bool curand_initialized = false;

// Lightweight metadata passed directly to CUDA threads
struct TensorMeta {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    size_t ndim;
    size_t storage_offset;
};

// Host function: Create TensorMeta to pass to kernels
TensorMeta create_tensor_meta(const TensorImpl *t) {
    TensorMeta meta;
    meta.ndim = t->ndim;
    meta.storage_offset = t->storage_offset;
    for (size_t i = 0; i < t->ndim && i < MAX_DIMS; ++i) {
        meta.shape[i] = t->shape[i];
        meta.strides[i] = t->strides[i];
    }
    return meta;
}

// Device function: Maps linear thread index (0..numel-1) -> Physical VRAM index
__device__ inline size_t get_physical_offset(size_t linear_idx, TensorMeta meta) {
    size_t offset = meta.storage_offset;
    for (int k = (int)meta.ndim - 1; k >= 0; --k) {
        size_t dim_idx = linear_idx % meta.shape[k];
        offset += dim_idx * meta.strides[k];
        linear_idx /= meta.shape[k];
    }
    return offset;
}

// ============================================================================
// MEMORY AND INITIALIZATION ROUTINES
// ============================================================================

void cyflow_manual_seed_cuda(unsigned long long seed) {
  if (!curand_initialized) {
    CURAND_CHECK(curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT));
    curand_initialized = true;
  }
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cuda_gen, seed));
}

void tensor_fill_uniform_cuda(TensorImpl *tensor) {
  if (!tensor || !tensor->storage || !tensor->storage->data)
    return;
  if (!curand_initialized) {
    cyflow_manual_seed_cuda(42ULL);
  }
  CURAND_CHECK(
      curandGenerateUniform(cuda_gen, tensor->storage->data, tensor->numel));
}

Storage *storage_create_cuda(size_t size) {
  Storage *storage = (Storage *)malloc(sizeof(Storage));
  if (!storage)
    return NULL;

  CUDA_CHECK(cudaMalloc((void **)&storage->data, size * sizeof(float)));
  CUDA_CHECK(cudaMemset(storage->data, 0, size * sizeof(float)));

  storage->size = size;
  storage->ref_count = 1;
  storage->owns_data = true;
  storage->device = DEVICE_CUDA;
  return storage;
}

void storage_free_cuda(Storage *storage) {
  if (!storage)
    return;
  storage->ref_count--;
  if (storage->ref_count == 0) {
    if (storage->owns_data && storage->data) {
      cudaFree(storage->data); 
    }
    free(storage);
  }
}

void tensor_free_cuda(TensorImpl *tensor){
  if(!tensor) return;
  if (tensor->storage){
    storage_free_cuda(tensor->storage);
  }
  if (tensor->shape) free(tensor->shape);
  if (tensor->strides) free(tensor->strides);
  free(tensor);
}

TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim) {
  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;

  tensor->ndim = ndim;
  tensor->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
  tensor->strides = (int64_t *)malloc(ndim * sizeof(int64_t));
  if (!tensor->shape || !tensor->strides) {
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return NULL;
  }
  memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
  compute_contiguous_strides(tensor->strides, tensor->shape, ndim); // Assuming this is defined elsewhere
  
  size_t total_size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    total_size *= shape[i];
  }
  tensor->numel = total_size;
  tensor->storage_offset = 0;

  tensor->storage = storage_create_cuda(total_size);
  if (!tensor->storage) {
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return NULL;
  }

  return tensor;
}

void tensor_set_data_cuda(TensorImpl *tensor, const float *data) {
  if (!tensor || !tensor->storage || !tensor->storage->data || !data)
    return;
  CUDA_CHECK(cudaMemcpy(tensor->storage->data, data,
                        tensor->numel * sizeof(float), cudaMemcpyHostToDevice));
}

bool tensor_is_contiguous_cuda(const TensorImpl *tensor) {
    if (!tensor || tensor->ndim == 0) return true;
    int64_t expected_stride = 1;
    for (int i = (int)tensor->ndim - 1; i >= 0; --i) {
        if (tensor->shape[i] != 1 && tensor->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape[i];
    }
    return true;
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

// --- Scalar Contiguous Kernels ---
__global__ void kernel_add_scalar_contiguous(float *data, float val, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        data[i] += val;
    }
}

__global__ void kernel_mul_scalar_contiguous(float *data, float val, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        data[i] *= val;
    }
}

// --- Scalar Strided Kernels ---
__global__ void kernel_add_scalar_strided(float *data, float val, size_t numel, TensorMeta meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        data[get_physical_offset(i, meta)] += val;
    }
}

__global__ void kernel_mul_scalar_strided(float *data, float val, size_t numel, TensorMeta meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        data[get_physical_offset(i, meta)] *= val;
    }
}

// --- Tensor Contiguous Kernels ---
__global__ void kernel_add_tensor_contiguous(float *dst, const float *src, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) dst[i] += src[i];
}

__global__ void kernel_sub_tensor_contiguous(float *dst, const float *src, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) dst[i] -= src[i];
}

__global__ void kernel_mul_tensor_contiguous(float *dst, const float *src, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) dst[i] *= src[i];
}

__global__ void kernel_div_tensor_contiguous(float *dst, const float *src, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) dst[i] /= src[i];
}

// --- Tensor Strided Kernels ---
__global__ void kernel_add_tensor_strided(float *dst, const float *src, size_t numel, TensorMeta dst_meta, TensorMeta src_meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        dst[get_physical_offset(i, dst_meta)] += src[get_physical_offset(i, src_meta)];
    }
}

__global__ void kernel_sub_tensor_strided(float *dst, const float *src, size_t numel, TensorMeta dst_meta, TensorMeta src_meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        dst[get_physical_offset(i, dst_meta)] -= src[get_physical_offset(i, src_meta)];
    }
}

__global__ void kernel_mul_tensor_strided(float *dst, const float *src, size_t numel, TensorMeta dst_meta, TensorMeta src_meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        dst[get_physical_offset(i, dst_meta)] *= src[get_physical_offset(i, src_meta)];
    }
}

__global__ void kernel_div_tensor_strided(float *dst, const float *src, size_t numel, TensorMeta dst_meta, TensorMeta src_meta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        dst[get_physical_offset(i, dst_meta)] /= src[get_physical_offset(i, src_meta)];
    }
}

// ============================================================================
// HOST LAUNCHERS
// ============================================================================

extern "C" {

void tensor_add_scalar_cuda(TensorImpl *dst, float val) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst)) {
        float *d_ptr = dst->storage->data + dst->storage_offset;
        kernel_add_scalar_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_ptr, val, numel);
    } else {
        TensorMeta meta = create_tensor_meta(dst);
        kernel_add_scalar_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(dst->storage->data, val, numel, meta);
    }
}

void tensor_sub_scalar_cuda(TensorImpl *dst, float val) {
    tensor_add_scalar_cuda(dst, -val);
}

void tensor_mul_scalar_cuda(TensorImpl *dst, float val) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst)) {
        float *d_ptr = dst->storage->data + dst->storage_offset;
        kernel_mul_scalar_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_ptr, val, numel);
    } else {
        TensorMeta meta = create_tensor_meta(dst);
        kernel_mul_scalar_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(dst->storage->data, val, numel, meta);
    }
}

void tensor_div_scalar_cuda(TensorImpl *dst, float val) {
    tensor_mul_scalar_cuda(dst, 1.0f / val);
}

void tensor_add_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst) && tensor_is_contiguous_cuda(src)) {
        float *d_dst = dst->storage->data + dst->storage_offset;
        const float *d_src = src->storage->data + src->storage_offset;
        kernel_add_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_dst, d_src, numel);
    } else {
        TensorMeta dst_meta = create_tensor_meta(dst);
        TensorMeta src_meta = create_tensor_meta(src);
        kernel_add_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            dst->storage->data, src->storage->data, numel, dst_meta, src_meta
        );
    }
}

void tensor_sub_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst) && tensor_is_contiguous_cuda(src)) {
        float *d_dst = dst->storage->data + dst->storage_offset;
        const float *d_src = src->storage->data + src->storage_offset;
        kernel_sub_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_dst, d_src, numel);
    } else {
        TensorMeta dst_meta = create_tensor_meta(dst);
        TensorMeta src_meta = create_tensor_meta(src);
        kernel_sub_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            dst->storage->data, src->storage->data, numel, dst_meta, src_meta
        );
    }
}

void tensor_mul_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst) && tensor_is_contiguous_cuda(src)) {
        float *d_dst = dst->storage->data + dst->storage_offset;
        const float *d_src = src->storage->data + src->storage_offset;
        kernel_mul_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_dst, d_src, numel);
    } else {
        TensorMeta dst_meta = create_tensor_meta(dst);
        TensorMeta src_meta = create_tensor_meta(src);
        kernel_mul_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            dst->storage->data, src->storage->data, numel, dst_meta, src_meta
        );
    }
}

void tensor_div_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
    size_t numel = dst->numel;
    if (numel == 0) return;
    int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    if (tensor_is_contiguous_cuda(dst) && tensor_is_contiguous_cuda(src)) {
        float *d_dst = dst->storage->data + dst->storage_offset;
        const float *d_src = src->storage->data + src->storage_offset;
        kernel_div_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_dst, d_src, numel);
    } else {
        TensorMeta dst_meta = create_tensor_meta(dst);
        TensorMeta src_meta = create_tensor_meta(src);
        kernel_div_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            dst->storage->data, src->storage->data, numel, dst_meta, src_meta
        );
    }
}

} // extern "C"