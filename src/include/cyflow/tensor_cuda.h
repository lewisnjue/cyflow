#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include "cyflow/common.h"
#include "cyflow/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

Storage *storage_create_cuda(size_t size);
void storage_free_cuda(Storage *storage);
TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim);
void tensor_free_cuda(TensorImpl *tensor);
void tensor_fill_uniform_cuda(TensorImpl *tensor);
void cyflow_manual_seed_cuda(unsigned long long seed);
void tensor_set_data_cuda(TensorImpl *tensor, const float *data);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_CUDA_H
