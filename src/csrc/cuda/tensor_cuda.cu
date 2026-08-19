#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"
#include <cuda_runtime.h>
#include "cyflow/tensor.h"
#include <curand.h>
#include <math.h>
#include <stdio.h>

static curandGenerator_t cuda_gen;
static bool curand_initialized = false;

__device__ inline size_t get_physical_offset(size_t linear_idx,
                                             TensorMeta meta) {
  size_t offset = meta.storage_offset;
  for (int k = (int)meta.ndim - 1; k >= 0; --k) {
    size_t dim_idx = linear_idx % meta.shape[k];
    offset += dim_idx * meta.strides[k];
    linear_idx /= meta.shape[k];
  }
  return offset;
}

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

  cudaError_t err = cudaMalloc((void **)&storage->data, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    free(storage);
    return NULL;
  }

  err = cudaMemset(storage->data, 0, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    cudaFree(storage->data);
    free(storage);
    return NULL;
  }

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

void tensor_free_cuda(TensorImpl *tensor) {
  if (!tensor)
    return;
  if (tensor->storage) {
    storage_free_cuda(tensor->storage);
  }
  if (tensor->shape)
    free(tensor->shape);
  if (tensor->strides)
    free(tensor->strides);
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
  compute_contiguous_strides(tensor->strides, tensor->shape, ndim);

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


TensorImpl *tensor_clone_cuda_contiguous(const TensorImpl *src) {
  if (!src || !src->storage || !src->storage->data)
    return NULL;

  TensorImpl *clone = tensor_create_cuda(src->shape, src->ndim);
  if (!clone)
    return NULL;

  CUDA_CHECK(cudaMemcpy(clone->storage->data, src->storage->data + src->storage_offset,
                        src->numel * sizeof(float), cudaMemcpyDeviceToDevice));
  clone->storage_offset = src->storage_offset;
  return clone;
}


__global__ void kernel_clone_cuda_strided(float *dst_data, const float *src_data, 
                                          size_t numel, TensorMeta src_meta, size_t dst_offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < numel; i += stride) {
        // Calculate where to read from the strided source
        size_t src_physical_offset = get_physical_offset(i, src_meta);
        
        // The destination is contiguous, so we map logical index 'i' directly
        dst_data[dst_offset + i] = src_data[src_physical_offset];
    }
}

extern "C" TensorImpl *tensor_clone_cuda_strided(const TensorImpl *src) {
  if (!src || !src->storage || !src->storage->data)
    return NULL;

  TensorImpl *clone = tensor_create_cuda(src->shape, src->ndim);
  if (!clone)
    return NULL;

  if (src->numel == 0) {
      return clone;
  }

  // Set up execution configuration
  int threads = 256;
  int blocks = (src->numel + threads - 1) / threads;
  
  // Package the strided metadata for the GPU
  TensorMeta src_meta = create_tensor_meta(src);

  // Launch the kernel
  kernel_clone_cuda_strided<<<blocks, threads>>>(
      clone->storage->data, 
      src->storage->data, 
      src->numel, 
      src_meta, 
      clone->storage_offset
  );
  
  // Wait for the copy to complete
  cudaDeviceSynchronize();

  return clone;
}

TensorImpl *tensor_clone_cuda(const TensorImpl *src){
  if (!src || !src->storage || !src->storage->data)
    return NULL;

  bool contiguous = tensor_is_contiguous(src);
  if (contiguous) {
    return tensor_clone_cuda_contiguous(src);
  } else {
    return tensor_clone_cuda_strided(src);
  }
}