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
