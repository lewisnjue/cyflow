#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H
#include "cyflow/common.h"
Storage *storage_create_cpu(size_t size);
void storage_free_cpu(Storage *storage);
TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim);
void tensor_free_cpu(TensorImpl *tensor);
void cyflow_manual_seed(unsigned int seed);
void tensor_fill_uniform_cpu(TensorImpl *tensor);
void cyflow_manual_seed_cuda(unsigned long long seed);
#endif // TENSOR_CPU_H