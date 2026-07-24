#include "cyflow/tensor.h"
#include "cyflow/common.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TensorMeta create_tensor_meta(const TensorImpl *t) {
  if (t->ndim > MAX_DIMS) {
    fprintf(stderr, "CUDA Error: Tensor ndim (%zu) exceeds MAX_DIMS (%d)\n",
            t->ndim, MAX_DIMS);
    exit(1);
  }
  TensorMeta meta;
  meta.ndim = t->ndim;
  meta.storage_offset = t->storage_offset;
  for (size_t i = 0; i < t->ndim; ++i) {
    meta.shape[i] = t->shape[i];
    meta.strides[i] = t->strides[i];
  }
  return meta;
}

bool tensor_is_contiguous(const TensorImpl *tensor) {
  if (!tensor || tensor->ndim == 0)
    return true;
  int64_t expected_stride = 1;
  for (int i = (int)tensor->ndim - 1; i >= 0; --i) {
    if (tensor->shape[i] != 1 && tensor->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= tensor->shape[i];
  }
  return true;
}

bool tensor_is_contiguous_cpu(const TensorImpl *tensor) {
  return tensor_is_contiguous(tensor);
}

bool tensor_is_contiguous_cuda(const TensorImpl *tensor) {
  return tensor_is_contiguous(tensor);
}

void compute_contiguous_strides(int64_t *strides,
                                const int64_t *shape,
                                size_t ndim) {
  if (ndim == 0)
    return;
  strides[ndim - 1] = 1;
  for (int i = (int)ndim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

TensorImpl *tensor_view(TensorImpl *src, const int64_t *shape,
                        size_t ndim) {
  if (!src || !src->storage)
    return NULL;

  size_t new_numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    new_numel *= shape[i];
  }

  if (new_numel != src->numel) {
    fprintf(stderr,
            "Cyflow Error: Cannot reshape tensor of size %zu into shape with "
            "size %zu\n",
            src->numel, new_numel);
    return NULL;
  }

  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;

  tensor->ndim = ndim;
  tensor->numel = new_numel;
  tensor->storage_offset = src->storage_offset;

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

  tensor->storage = src->storage;
  tensor->storage->ref_count++;

  return tensor;
}

TensorImpl *tensor_index(TensorImpl *src, int64_t index) {
  if (!src || !src->storage)
    return NULL;
  size_t ndim = src->ndim;
  if (ndim == 0) {
    return NULL;
  }
  int64_t dim0 = src->shape[0];
  if (index < 0)
    index += dim0;
  if (index < 0 || index >= dim0) {
    fprintf(stderr, "Index %lld out of range for dimension size %zu\n",
            (long long)index, dim0);
    return NULL;
  }
  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;
  tensor->storage = src->storage;
  tensor->storage->ref_count++;
  tensor->storage_offset = src->storage_offset + index * src->strides[0];
  if (ndim > 1) {
    tensor->ndim = ndim - 1;
    tensor->numel = src->numel / dim0;
    tensor->shape = (int64_t *)malloc((ndim - 1) * sizeof(int64_t));
    tensor->strides = (int64_t *)malloc((ndim - 1) * sizeof(int64_t));
    for (size_t i = 1; i < ndim; ++i) {
      tensor->shape[i - 1] = src->shape[i];
      tensor->strides[i - 1] = src->strides[i];
    }
  } else {
    tensor->ndim = 0;
    tensor->numel = 1;
    tensor->shape = NULL;
    tensor->strides = NULL;
  }
  return tensor;
}

static uint32_t g_rng_state = 123456789;

Storage *storage_create_cpu(size_t size) {
  Storage *storage = (Storage *)malloc(sizeof(Storage));
  if (!storage)
    return NULL;

  storage->data = (float *)calloc(size, sizeof(float));
  if (!storage->data && size > 0) {
    free(storage);
    return NULL;
  }

  storage->size = size;
  storage->ref_count = 1;
  storage->owns_data = true;
  storage->device = DEVICE_CPU;
  return storage;
}

void storage_free_cpu(Storage *storage) {
  if (!storage)
    return;

  storage->ref_count--;
  if (storage->ref_count == 0) {
    if (storage->owns_data && storage->data) {
      free(storage->data);
    }
    free(storage);
  }
}

TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim) {
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
  compute_contiguous_strides(tensor->strides, tensor->shape,
                             ndim); // Assumes implemented elsewhere

  size_t total_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }
  tensor->numel = total_size;
  tensor->storage_offset = 0;

  tensor->storage = storage_create_cpu(total_size);
  if (!tensor->storage) {
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return NULL;
  }
  return tensor;
}

void tensor_free_cpu(TensorImpl *tensor) {
  if (!tensor)
    return;
  if (tensor->storage) {
    storage_free_cpu(tensor->storage);
  }
  if (tensor->shape)
    free(tensor->shape);
  if (tensor->strides)
    free(tensor->strides);
  free(tensor);
}

void tensor_fill_uniform_cpu(TensorImpl *tensor) {
  if (!tensor || !tensor->storage || !tensor->storage->data)
    return;

  float *data = tensor->storage->data + tensor->storage_offset;
  size_t numel = tensor->numel;

  // Pre-load state locally to help the compiler optimize the loop register
  uint32_t state = g_rng_state;

  // Multiplier to map uint32 [0, 4294967295] to float [0.0, 1.0)
  // 1.0f / 4294967296.0f
  const float scale = 2.3283064365386963e-10f;

  for (size_t i = 0; i < numel; i++) {
    // Inline xorshift step
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;

    // Convert to float via multiplication instead of division
    data[i] = (float)x * scale;
  }

  g_rng_state = state;
}

void cyflow_manual_seed_cpu(unsigned long long seed) {
  // Ensure state is never 0 (which breaks Xorshift)
  g_rng_state = (uint32_t)(seed ^ (seed >> 32));
  if (g_rng_state == 0) {
    g_rng_state = 1;
  }
}

static inline uint32_t xorshift32(void) {
  uint32_t x = g_rng_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  g_rng_state = x;
  return x;
}

void tensor_set_data_cpu(TensorImpl *tensor, const float *data) {
  if (!tensor || !tensor->storage || !tensor->storage->data || !data)
    return;
  // This assumes the incoming data buffer exactly matches the memory layout.
  memcpy(tensor->storage->data + tensor->storage_offset, data,
         tensor->numel * sizeof(float));
}
