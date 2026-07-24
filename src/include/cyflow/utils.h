#ifndef CYFLOW_UTILS_H
#define CYFLOW_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include "cyflow/tensor.h" // Required for TensorImpl

#ifdef __cplusplus
extern "C" {
#endif

int compute_broadcast_shape(const int64_t *shape_a, size_t ndim_a,
                            const int64_t *shape_b, size_t ndim_b,
                            int64_t *out_shape, size_t *out_ndim);

// Add the CPU unbroadcast declaration
TensorImpl* tensor_unbroadcast_cpu(const TensorImpl* grad, const int64_t* target_shape, size_t target_ndim);

// Add the CUDA unbroadcast declaration
TensorImpl* tensor_unbroadcast_cuda(const TensorImpl* grad, const int64_t* target_shape, size_t target_ndim);

#ifdef __cplusplus
}
#endif

#endif // CYFLOW_UTILS_H