#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#include "cyflow/common.h"
#include "cyflow/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void tensor_matmul_out_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_CUDA_H
