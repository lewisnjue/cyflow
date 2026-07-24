#ifndef OUT_OP_CUDA_H
#define OUT_OP_CUDA_H

#include "cyflow/common.h"
#include "cyflow/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void tensor_add_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_sub_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_mul_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_div_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val);

void tensor_add_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_sub_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_mul_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val);
void tensor_div_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val);


void tensor_add_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_sub_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_mul_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_div_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);

void tensor_add_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_sub_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_mul_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);
void tensor_div_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);

#ifdef __cplusplus
}
#endif

#endif // OUT_OP_CUDA_H