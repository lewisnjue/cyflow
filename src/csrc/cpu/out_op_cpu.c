#include "cyflow/out_op_cpu.h"
#include <stdlib.h>

#define CPU_OUT_SCALAR_OP_CONTIGUOUS(dst, src, val, op)                        \
  float *d_ptr = dst->storage->data + dst->storage_offset;                     \
  const float *s_ptr = src->storage->data + src->storage_offset;               \
  size_t numel = dst->numel;                                                   \
  _Pragma("omp parallel for simd") for (size_t i = 0; i < numel; ++i) {        \
    d_ptr[i] = s_ptr[i] op val;                                                \
  }

// Macro for C = A + val (where A is sliced/transposed)
#define CPU_OUT_SCALAR_OP_STRIDED(dst, src, val, op)                           \
  float *d_ptr = dst->storage->data + dst->storage_offset;                     \
  const float *s_data = src->storage->data;                                    \
  size_t numel = dst->numel;                                                   \
  size_t ndim = dst->ndim;                                                     \
  size_t s_base = src->storage_offset;                                         \
  _Pragma("omp parallel for") for (size_t i = 0; i < numel; ++i) {             \
    size_t curr = i;                                                           \
    size_t s_phys = s_base;                                                    \
    for (int k = (int)ndim - 1; k >= 0; --k) {                                 \
      size_t dim_idx = curr % dst->shape[k];                                   \
      s_phys += dim_idx * src->strides[k];                                     \
      curr /= dst->shape[k];                                                   \
    }                                                                          \
    d_ptr[i] = s_data[s_phys] op val;                                          \
  }

// Macro for C = A + B (where A and B are identical shape and contiguous)
#define CPU_OUT_TENSOR_OP_CONTIGUOUS(dst, src1, src2, op)                      \
  float *d_ptr = dst->storage->data + dst->storage_offset;                     \
  const float *s1_ptr = src1->storage->data + src1->storage_offset;            \
  const float *s2_ptr = src2->storage->data + src2->storage_offset;            \
  size_t numel = dst->numel;                                                   \
  _Pragma("omp parallel for simd") for (size_t i = 0; i < numel; ++i) {        \
    d_ptr[i] = s1_ptr[i] op s2_ptr[i];                                         \
  }

// Macro for C = A + B (with broadcasting and arbitrary strides)
// Note: We use C99 Variable Length Arrays (VLA) for stride1 and stride2 
// to avoid slow malloc calls inside the macro.
#define CPU_OUT_TENSOR_OP_STRIDED(dst, src1, src2, op)                         \
  size_t ndim = dst->ndim;                                                     \
  size_t numel = dst->numel;                                                   \
  int64_t st1[ndim]; /* C99 VLA on the stack */                                \
  int64_t st2[ndim];                                                           \
  for (int d = (int)ndim - 1; d >= 0; --d) {                                   \
      int s1_idx = d - (ndim - src1->ndim);                                    \
      st1[d] = (s1_idx >= 0 && src1->shape[s1_idx] != 1) ? src1->strides[s1_idx] : 0; \
      int s2_idx = d - (ndim - src2->ndim);                                    \
      st2[d] = (s2_idx >= 0 && src2->shape[s2_idx] != 1) ? src2->strides[s2_idx] : 0; \
  }                                                                            \
  float *d_ptr = dst->storage->data + dst->storage_offset;                     \
  const float *s1_data = src1->storage->data + src1->storage_offset;           \
  const float *s2_data = src2->storage->data + src2->storage_offset;           \
  _Pragma("omp parallel for") for (size_t i = 0; i < numel; ++i) {             \
    size_t curr = i;                                                           \
    size_t off1 = 0, off2 = 0;                                                 \
    for (int k = (int)ndim - 1; k >= 0; --k) {                                 \
      size_t coord = curr % dst->shape[k];                                     \
      off1 += coord * st1[k];                                                  \
      off2 += coord * st2[k];                                                  \
      curr /= dst->shape[k];                                                   \
    }                                                                          \
    d_ptr[i] = s1_data[off1] op s2_data[off2];                                 \
  }




// Contiguous
void tensor_add_out_scalar_contiguous_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_CONTIGUOUS(dst, src, val, +);
}
void tensor_sub_out_scalar_contiguous_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_CONTIGUOUS(dst, src, val, -);
}
void tensor_mul_out_scalar_contiguous_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_CONTIGUOUS(dst, src, val, *);
}
void tensor_div_out_scalar_contiguous_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_CONTIGUOUS(dst, src, val, /);
}

// Strided
void tensor_add_out_scalar_strided_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_STRIDED(dst, src, val, +);
}
void tensor_sub_out_scalar_strided_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_STRIDED(dst, src, val, -);
}
void tensor_mul_out_scalar_strided_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_STRIDED(dst, src, val, *);
}
void tensor_div_out_scalar_strided_cpu(TensorImpl *dst, const TensorImpl *src, float val) {
    CPU_OUT_SCALAR_OP_STRIDED(dst, src, val, /);
}

// ==========================================
// TENSOR IMPLEMENTATIONS
// ==========================================

// Contiguous
void tensor_add_out_tensor_contiguous_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_CONTIGUOUS(dst, src1, src2, +);
}
void tensor_sub_out_tensor_contiguous_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_CONTIGUOUS(dst, src1, src2, -);
}
void tensor_mul_out_tensor_contiguous_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_CONTIGUOUS(dst, src1, src2, *);
}
void tensor_div_out_tensor_contiguous_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_CONTIGUOUS(dst, src1, src2, /);
}

// Strided
void tensor_add_out_tensor_strided_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_STRIDED(dst, src1, src2, +);
}
void tensor_sub_out_tensor_strided_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_STRIDED(dst, src1, src2, -);
}
void tensor_mul_out_tensor_strided_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_STRIDED(dst, src1, src2, *);
}
void tensor_div_out_tensor_strided_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    CPU_OUT_TENSOR_OP_STRIDED(dst, src1, src2, /);
}