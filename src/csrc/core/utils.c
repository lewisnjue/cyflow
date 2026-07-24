#include "cyflow/utils.h"
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

int compute_broadcast_shape(
    const int64_t *shape_a, size_t ndim_a,
    const int64_t *shape_b, size_t ndim_b,
    int64_t *out_shape, size_t *out_ndim) 
{
    size_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    for (size_t i = 1; i <= max_ndim; i++) {
        int64_t dim_a = (i <= ndim_a) ? shape_a[ndim_a - i] : 1;
        int64_t dim_b = (i <= ndim_b) ? shape_b[ndim_b - i] : 1;

        if (dim_a == dim_b) {
            out_shape[max_ndim - i] = dim_a;
        } else if (dim_a == 1) {
            out_shape[max_ndim - i] = dim_b;
        } else if (dim_b == 1) {
            out_shape[max_ndim - i] = dim_a;
        } else {
            return -1; 
        }
    }
    return 0; 
}

TensorImpl* tensor_unbroadcast_cpu(const TensorImpl* grad, const int64_t* target_shape, size_t target_ndim) {
    TensorImpl* result = tensor_create_cpu(target_shape, target_ndim);
    if (!result) return NULL;
    
    memset(result->storage->data, 0, result->storage->size * sizeof(float));
    if (grad->numel == 0) return result;

    int64_t* grad_coords = (int64_t*)malloc(grad->ndim * sizeof(int64_t));
    int64_t* target_coords = (int64_t*)malloc(target_ndim * sizeof(int64_t));
    size_t ndim_diff = grad->ndim - target_ndim;

    for (size_t i = 0; i < grad->numel; i++) {
        size_t temp = i;
        
        for (ptrdiff_t d = grad->ndim - 1; d >= 0; d--) {
            grad_coords[d] = temp % grad->shape[d];
            temp /= grad->shape[d];
        }

        for (size_t d = 0; d < target_ndim; d++) {
            if (target_shape[d] == 1) {
                target_coords[d] = 0; 
            } else {
                target_coords[d] = grad_coords[ndim_diff + d];
            }
        }

        size_t grad_offset = grad->storage_offset;
        for (size_t d = 0; d < grad->ndim; d++) {
            grad_offset += grad_coords[d] * grad->strides[d];
        }

        size_t target_offset = result->storage_offset;
        for (size_t d = 0; d < target_ndim; d++) {
            target_offset += target_coords[d] * result->strides[d];
        }

        result->storage->data[target_offset] += grad->storage->data[grad_offset];
    }

    free(grad_coords);
    free(target_coords);

    return result;
}