#ifndef CYFLOW_INLINE_OP_CUDA_H
#define CYFLOW_INLINE_OP_CUDA_H

#include "cyflow/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Host-callable CUDA functions (called from C++, Cython, etc.) */
void tensor_add_scalar_cuda(TensorImpl *dst, float val);
void tensor_mul_scalar_cuda(TensorImpl *dst, float val);
void tensor_add_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_sub_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_mul_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_div_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_div_scalar_cuda(TensorImpl *dst, float val);
void tensor_sub_scalar_cuda(TensorImpl *dst, float val);
#ifdef __cplusplus
}
#endif

/* CUDA device-only functions visible ONLY to nvcc */
#ifdef __CUDACC__
__device__ size_t get_physical_offset(size_t linear_idx, TensorMeta meta);
#endif

#endif