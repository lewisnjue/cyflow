#ifndef INLINE_OP_CPU_H
#define INLINE_OP_CPU_H

#include "cyflow/common.h"
#include "cyflow/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CPU_SCALAR_OP_CONTIGUOUS(dst, val, op)                                 \
  float *ptr = dst->storage->data + dst->storage_offset;                       \
  size_t numel = dst->numel;                                                   \
  _Pragma("omp parallel for simd") for (size_t i = 0; i < numel; ++i) {        \
    ptr[i] op val;                                                             \
  }

// Macro helper for strided scalar operations on non-contiguous views
#define CPU_SCALAR_OP_STRIDED(dst, val, op)                                    \
  float *data = dst->storage->data;                                            \
  size_t numel = dst->numel;                                                   \
  size_t ndim = dst->ndim;                                                     \
  size_t base_offset = dst->storage_offset;                                    \
  _Pragma("omp parallel for") for (size_t i = 0; i < numel; ++i) {             \
    size_t curr = i;                                                           \
    size_t phys_offset = base_offset;                                          \
    for (int k = (int)ndim - 1; k >= 0; --k) {                                 \
      size_t dim_idx = curr % dst->shape[k];                                   \
      phys_offset += dim_idx * dst->strides[k];                                \
      curr /= dst->shape[k];                                                   \
    }                                                                          \
    data[phys_offset] op val;                                                  \
  }

// Macro helper for contiguous tensor-tensor operations
#define CPU_TENSOR_OP_CONTIGUOUS(dst, src, op)                                 \
  float *d_ptr = dst->storage->data + dst->storage_offset;                     \
  const float *s_ptr = src->storage->data + src->storage_offset;               \
  size_t numel = dst->numel;                                                   \
  _Pragma("omp parallel for simd") for (size_t i = 0; i < numel; ++i) {        \
    d_ptr[i] op s_ptr[i];                                                      \
  }

// Macro helper for strided tensor-tensor operations on non-contiguous views
#define CPU_TENSOR_OP_STRIDED(dst, src, op)                                    \
  float *dst_data = dst->storage->data;                                        \
  const float *src_data = src->storage->data;                                  \
  size_t numel = dst->numel;                                                   \
  size_t ndim = dst->ndim;                                                     \
  size_t dst_base = dst->storage_offset;                                       \
  size_t src_base = src->storage_offset;                                       \
  _Pragma("omp parallel for") for (size_t i = 0; i < numel; ++i) {             \
    size_t curr = i;                                                           \
    size_t dst_phys = dst_base;                                                \
    size_t src_phys = src_base;                                                \
    for (int k = (int)ndim - 1; k >= 0; --k) {                                 \
      size_t dim_idx = curr % dst->shape[k];                                   \
      dst_phys += dim_idx * dst->strides[k];                                   \
      src_phys += dim_idx * src->strides[k];                                   \
      curr /= dst->shape[k];                                                   \
    }                                                                          \
    dst_data[dst_phys] op src_data[src_phys];                                  \
  }


void tensor_add_scalar_cpu(TensorImpl *dst, float val);
void tensor_sub_scalar_cpu(TensorImpl *dst, float val);
void tensor_mul_scalar_cpu(TensorImpl *dst, float val);
void tensor_div_scalar_cpu(TensorImpl *dst, float val);
void tensor_add_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_sub_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_mul_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_div_tensor_cpu(TensorImpl *dst, const TensorImpl *src);

#ifdef __cplusplus
}
#endif

#endif // INLINE_OP_CPU_H
