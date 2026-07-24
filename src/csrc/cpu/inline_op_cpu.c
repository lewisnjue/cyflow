#include "cyflow/inline_op_cpu.h"
#include "cyflow/common.h"
#include "cyflow/tensor.h"
#include <stdbool.h>

void tensor_add_scalar_cpu(TensorImpl *dst, float val) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst)) {
    CPU_SCALAR_OP_CONTIGUOUS(dst, val, +=);
  } else {
    CPU_SCALAR_OP_STRIDED(dst, val, +=);
  }
}

void tensor_sub_scalar_cpu(TensorImpl *dst, float val) {
  tensor_add_scalar_cpu(dst, -val);
}

void tensor_mul_scalar_cpu(TensorImpl *dst, float val) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst)) {
    CPU_SCALAR_OP_CONTIGUOUS(dst, val, *=);
  } else {
    CPU_SCALAR_OP_STRIDED(dst, val, *=);
  }
}

void tensor_div_scalar_cpu(TensorImpl *dst, float val) {
  tensor_mul_scalar_cpu(dst, 1.0f / val);
}

void tensor_add_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    CPU_TENSOR_OP_CONTIGUOUS(dst, src, +=);
  } else {
    CPU_TENSOR_OP_STRIDED(dst, src, +=);
  }
}

void tensor_sub_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    CPU_TENSOR_OP_CONTIGUOUS(dst, src, -=);
  } else {
    CPU_TENSOR_OP_STRIDED(dst, src, -=);
  }
}

void tensor_mul_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    CPU_TENSOR_OP_CONTIGUOUS(dst, src, *=);
  } else {
    CPU_TENSOR_OP_STRIDED(dst, src, *=);
  }
}

void tensor_div_tensor_cpu(TensorImpl *dst, const TensorImpl *src) {
  if (dst->numel == 0)
    return;
  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    CPU_TENSOR_OP_CONTIGUOUS(dst, src, /=);
  } else {
    CPU_TENSOR_OP_STRIDED(dst, src, /=);
  }
}
