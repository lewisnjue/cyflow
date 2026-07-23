#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include "cyflow/common.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MEMORY AND INITIALIZATION ROUTINES
// ============================================================================
Storage *storage_create_cuda(size_t size);
void storage_free_cuda(Storage *storage);
TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim);
void tensor_free_cuda(TensorImpl *tensor);

void tensor_fill_uniform_cuda(TensorImpl *tensor);
void cyflow_manual_seed_cuda(unsigned long long seed);
void tensor_set_data_cuda(TensorImpl *tensor, const float *data);

// ============================================================================
// UTILITIES
// ============================================================================
bool tensor_is_contiguous_cuda(const TensorImpl *tensor);

// ============================================================================
// IN-PLACE SCALAR OPERATIONS
// ============================================================================
void tensor_add_scalar_cuda(TensorImpl *dst, float val);
void tensor_sub_scalar_cuda(TensorImpl *dst, float val);
void tensor_mul_scalar_cuda(TensorImpl *dst, float val);
void tensor_div_scalar_cuda(TensorImpl *dst, float val);

// ============================================================================
// IN-PLACE TENSOR OPERATIONS
// ============================================================================
void tensor_add_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_sub_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_mul_tensor_cuda(TensorImpl *dst, const TensorImpl *src);
void tensor_div_tensor_cuda(TensorImpl *dst, const TensorImpl *src);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_CUDA_H