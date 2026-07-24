#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H

#include "cyflow/common.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MEMORY AND INITIALIZATION ROUTINES
// ============================================================================
Storage *storage_create_cpu(size_t size);
void storage_free_cpu(Storage *storage);

TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim);
void tensor_free_cpu(TensorImpl *tensor);

void cyflow_manual_seed_cpu(unsigned long long seed);
void tensor_fill_uniform_cpu(TensorImpl *tensor);
void tensor_set_data_cpu(TensorImpl *tensor, const float *data);

// ============================================================================
// UTILITIES
// ============================================================================
bool tensor_is_contiguous_cpu(const TensorImpl *tensor);

// ============================================================================
// IN-PLACE SCALAR OPERATIONS
// ============================================================================
void tensor_add_scalar_cpu(TensorImpl *dst, float val);
void tensor_sub_scalar_cpu(TensorImpl *dst, float val);
void tensor_mul_scalar_cpu(TensorImpl *dst, float val);
void tensor_div_scalar_cpu(TensorImpl *dst, float val);

// ============================================================================
// IN-PLACE TENSOR OPERATIONS
// ============================================================================
void tensor_add_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_sub_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_mul_tensor_cpu(TensorImpl *dst, const TensorImpl *src);
void tensor_div_tensor_cpu(TensorImpl *dst, const TensorImpl *src);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_CPU_H
