#include "cyflow/tensor_cpu.h"
#include "cyflow/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ============================================================================
// MACROS FOR OPERATIONS
// ============================================================================

// Macro helper for contiguous scalar operations with OpenMP multithreading & SIMD
#define CPU_SCALAR_OP_CONTIGUOUS(dst, val, op)                      \
    float *ptr = dst->storage->data + dst->storage_offset;          \
    size_t numel = dst->numel;                                      \
    _Pragma("omp parallel for simd")                                \
    for (size_t i = 0; i < numel; ++i) {                            \
        ptr[i] op val;                                             \
    }

// Macro helper for strided scalar operations on non-contiguous views
#define CPU_SCALAR_OP_STRIDED(dst, val, op)                               \
    float *data = dst->storage->data;                                     \
    size_t numel = dst->numel;                                            \
    size_t ndim = dst->ndim;                                              \
    size_t base_offset = dst->storage_offset;                             \
    _Pragma("omp parallel for")                                           \
    for (size_t i = 0; i < numel; ++i) {                                  \
        size_t curr = i;                                                  \
        size_t phys_offset = base_offset;                                 \
        for (int k = (int)ndim - 1; k >= 0; --k) {                        \
            size_t dim_idx = curr % dst->shape[k];                        \
            phys_offset += dim_idx * dst->strides[k];                     \
            curr /= dst->shape[k];                                        \
        }                                                                 \
        data[phys_offset] op val;                                         \
    }

// Macro helper for contiguous tensor-tensor operations
#define CPU_TENSOR_OP_CONTIGUOUS(dst, src, op)                      \
    float *d_ptr = dst->storage->data + dst->storage_offset;        \
    const float *s_ptr = src->storage->data + src->storage_offset;  \
    size_t numel = dst->numel;                                      \
    _Pragma("omp parallel for simd")                                \
    for (size_t i = 0; i < numel; ++i) {                            \
        d_ptr[i] op s_ptr[i];                                       \
    }

// Macro helper for strided tensor-tensor operations on non-contiguous views
#define CPU_TENSOR_OP_STRIDED(dst, src, op)                               \
    float *dst_data = dst->storage->data;                                 \
    const float *src_data = src->storage->data;                           \
    size_t numel = dst->numel;                                            \
    size_t ndim = dst->ndim;                                              \
    size_t dst_base = dst->storage_offset;                                \
    size_t src_base = src->storage_offset;                                \
    _Pragma("omp parallel for")                                           \
    for (size_t i = 0; i < numel; ++i) {                                  \
        size_t curr = i;                                                  \
        size_t dst_phys = dst_base;                                       \
        size_t src_phys = src_base;                                       \
        for (int k = (int)ndim - 1; k >= 0; --k) {                        \
            size_t dim_idx = curr % dst->shape[k];                        \
            dst_phys += dim_idx * dst->strides[k];                        \
            src_phys += dim_idx * src->strides[k];                        \
            curr /= dst->shape[k];                                        \
        }                                                                 \
        dst_data[dst_phys] op src_data[src_phys];                         \
    }

// ============================================================================
// MEMORY AND INITIALIZATION ROUTINES
// ============================================================================

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
  compute_contiguous_strides(tensor->strides, tensor->shape, ndim); // Assumes implemented elsewhere
  
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

  for (size_t i = 0; i < numel; i++) {
    data[i] = (float)rand() / (float)RAND_MAX;
  }
}

void cyflow_manual_seed(unsigned int seed) { srand(seed); }

void tensor_set_data_cpu(TensorImpl *tensor, const float *data) {
  if (!tensor || !tensor->storage || !tensor->storage->data || !data)
    return;
  // This assumes the incoming data buffer exactly matches the memory layout.
  memcpy(tensor->storage->data + tensor->storage_offset, data, tensor->numel * sizeof(float));
}

// ============================================================================
// UTILITIES
// ============================================================================

bool tensor_is_contiguous_cpu(const TensorImpl *tensor) {
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
// IN-PLACE SCALAR OPERATIONS
// ============================================================================

void tensor_add_scalar_cpu(TensorImpl *dst, float val) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst)) {
        CPU_SCALAR_OP_CONTIGUOUS(dst, val, +=);
    } else {
        CPU_SCALAR_OP_STRIDED(dst, val, +=);
    }
}

void tensor_sub_scalar_cpu(TensorImpl *dst, float val) {
    tensor_add_scalar_cpu(dst, -val);
}

void tensor_mul_scalar_cpu(TensorImpl *dst, float val) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst)) {
        CPU_SCALAR_OP_CONTIGUOUS(dst, val, *=);
    } else {
        CPU_SCALAR_OP_STRIDED(dst, val, *=);
    }
}

void tensor_div_scalar_cpu(TensorImpl *dst, float val) {
    tensor_mul_scalar_cpu(dst, 1.0f / val);
}

// ============================================================================
// IN-PLACE TENSOR OPERATIONS
// ============================================================================

void tensor_add_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst) && tensor_is_contiguous_cpu(src)) {
        CPU_TENSOR_OP_CONTIGUOUS(dst, src, +=);
    } else {
        CPU_TENSOR_OP_STRIDED(dst, src, +=);
    }
}

void tensor_sub_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst) && tensor_is_contiguous_cpu(src)) {
        CPU_TENSOR_OP_CONTIGUOUS(dst, src, -=);
    } else {
        CPU_TENSOR_OP_STRIDED(dst, src, -=);
    }
}

void tensor_mul_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst) && tensor_is_contiguous_cpu(src)) {
        CPU_TENSOR_OP_CONTIGUOUS(dst, src, *=);
    } else {
        CPU_TENSOR_OP_STRIDED(dst, src, *=);
    }
}

void tensor_div_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
    if (dst->numel == 0) return;
    if (tensor_is_contiguous_cpu(dst) && tensor_is_contiguous_cpu(src)) {
        CPU_TENSOR_OP_CONTIGUOUS(dst, src, /=);
    } else {
        CPU_TENSOR_OP_STRIDED(dst, src, /=);
    }
}