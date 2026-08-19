#include "cyflow/utils.h"
#include <cuda_runtime.h>
#include "cyflow/tensor_cuda.h"
__global__ void unbroadcast_cuda_kernel(
    const float* grad_data,
    const int64_t* grad_shape,
    const int64_t* grad_strides,
    size_t grad_ndim,
    float* target_data,
    const int64_t* target_shape,
    const int64_t* target_strides,
    size_t target_ndim,
    size_t numel,
    size_t ndim_diff,
    size_t grad_base_offset,
    size_t target_base_offset
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numel) return;

    size_t temp = idx;
    size_t grad_offset = grad_base_offset;
    size_t target_offset = target_base_offset;

    for (ptrdiff_t d = grad_ndim - 1; d >= 0; d--) {
        int64_t g_coord = temp % grad_shape[d];
        temp /= grad_shape[d];
        
        grad_offset += g_coord * grad_strides[d];
        
        if (d >= ndim_diff) {
            size_t t_d = d - ndim_diff;
            int64_t t_coord = (target_shape[t_d] == 1) ? 0 : g_coord;
            target_offset += t_coord * target_strides[t_d];
        }
    }
    
    atomicAdd(&target_data[target_offset], grad_data[grad_offset]);
}

extern "C" TensorImpl* tensor_unbroadcast_cuda(const TensorImpl* grad, const int64_t* target_shape, size_t target_ndim) {
    TensorImpl* result = tensor_create_cuda(target_shape, target_ndim);
    if (!result) return NULL;
    
    cudaMemset(result->storage->data, 0, result->storage->size * sizeof(float));
    if (grad->numel == 0) return result;

    int64_t *d_grad_shape, *d_grad_strides, *d_target_shape, *d_target_strides;
    
    cudaMalloc(&d_grad_shape, grad->ndim * sizeof(int64_t));
    cudaMalloc(&d_grad_strides, grad->ndim * sizeof(int64_t));
    cudaMalloc(&d_target_shape, target_ndim * sizeof(int64_t));
    cudaMalloc(&d_target_strides, target_ndim * sizeof(int64_t));
    
    cudaMemcpy(d_grad_shape, grad->shape, grad->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_strides, grad->strides, grad->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_shape, result->shape, target_ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_strides, result->strides, target_ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    size_t ndim_diff = (grad->ndim >= target_ndim) ? (grad->ndim - target_ndim) : 0;
    
    int threads = 256;
    int blocks = (grad->numel + threads - 1) / threads;
    
    unbroadcast_cuda_kernel<<<blocks, threads>>>(
        grad->storage->data, d_grad_shape, d_grad_strides, grad->ndim,
        result->storage->data, d_target_shape, d_target_strides, target_ndim,
        grad->numel, ndim_diff, grad->storage_offset, result->storage_offset
    );
    
    cudaDeviceSynchronize();
    
    cudaFree(d_grad_shape);
    cudaFree(d_grad_strides);
    cudaFree(d_target_shape);
    cudaFree(d_target_strides);
    
    return result; 
}

__global__ void accumulate_view_grad_cuda_kernel(
    const float* grad_data,
    const int64_t* grad_shape,
    const int64_t* grad_strides,
    size_t grad_ndim,
    size_t grad_base_offset,
    const int64_t* view_shape,
    const int64_t* view_strides,
    size_t view_ndim,
    size_t view_base_offset,
    const int64_t* parent_shape,
    const int64_t* parent_strides,
    size_t parent_ndim,
    size_t parent_base_offset,
    float* result_data,
    const int64_t* result_strides,
    size_t result_base_offset,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    size_t temp = idx;
    size_t grad_storage_offset = grad_base_offset;
    for (ptrdiff_t d = (ptrdiff_t)grad_ndim - 1; d >= 0; d--) {
        int64_t coord = temp % grad_shape[d];
        temp /= grad_shape[d];
        grad_storage_offset += (size_t)(coord * grad_strides[d]);
    }

    temp = idx;
    size_t view_offset = view_base_offset;
    for (ptrdiff_t d = (ptrdiff_t)view_ndim - 1; d >= 0; d--) {
        int64_t coord = temp % view_shape[d];
        temp /= view_shape[d];
        view_offset += (size_t)(coord * view_strides[d]);
    }

    // Decode the raw storage offset into parent coordinates by greedily
    // consuming the largest *remaining* stride first (matches CPU logic).
    size_t rel = view_offset - parent_base_offset;
    size_t remaining = rel;

    int64_t parent_coords[MAX_DIMS];
    bool used[MAX_DIMS];
    for (size_t i = 0; i < (size_t)parent_ndim && i < MAX_DIMS; i++) used[i] = false;

    for (size_t step = 0; step < parent_ndim; step++) {
        int64_t max_stride = -1;
        int best_d = -1;
        for (size_t d = 0; d < parent_ndim; d++) {
            if (!used[d] && parent_strides[d] > max_stride) {
                max_stride = parent_strides[d];
                best_d = (int)d;
            }
        }
        if (best_d == -1) break;
        used[best_d] = true;

        if (parent_shape[best_d] == 0 || max_stride == 0) {
            parent_coords[best_d] = 0;
            continue;
        }
        parent_coords[best_d] = (int64_t)(remaining / (size_t)max_stride);
        remaining %= (size_t)max_stride;
    }

    size_t result_offset = result_base_offset;
    for (size_t d = 0; d < parent_ndim; d++) {
        result_offset += (size_t)(parent_coords[d] * result_strides[d]);
    }

    atomicAdd(&result_data[result_offset], grad_data[grad_storage_offset]);
}

extern "C" TensorImpl* tensor_accumulate_view_grad_cuda(
    const TensorImpl* parent,
    const TensorImpl* view,
    const TensorImpl* grad
) {
    TensorImpl* result = tensor_create_cuda(parent->shape, parent->ndim);
    if (!result) return NULL;

    cudaMemset(result->storage->data, 0, result->storage->size * sizeof(float));
    if (!view || !grad || grad->numel == 0) return result;

    int64_t *d_grad_shape, *d_grad_strides;
    int64_t *d_view_shape, *d_view_strides, *d_parent_shape, *d_parent_strides, *d_result_strides;

    cudaMalloc(&d_grad_shape, grad->ndim * sizeof(int64_t));
    cudaMalloc(&d_grad_strides, grad->ndim * sizeof(int64_t));
    cudaMalloc(&d_view_shape, view->ndim * sizeof(int64_t));
    cudaMalloc(&d_view_strides, view->ndim * sizeof(int64_t));
    cudaMalloc(&d_parent_shape, parent->ndim * sizeof(int64_t));
    cudaMalloc(&d_parent_strides, parent->ndim * sizeof(int64_t));
    cudaMalloc(&d_result_strides, parent->ndim * sizeof(int64_t));

    cudaMemcpy(d_grad_shape, grad->shape, grad->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_strides, grad->strides, grad->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_view_shape, view->shape, view->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_view_strides, view->strides, view->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent_shape, parent->shape, parent->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent_strides, parent->strides, parent->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_strides, result->strides, parent->ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (grad->numel + threads - 1) / threads;

    accumulate_view_grad_cuda_kernel<<<blocks, threads>>>(
        grad->storage->data,
        d_grad_shape,
        d_grad_strides,
        grad->ndim,
        grad->storage_offset,
        d_view_shape,
        d_view_strides,
        view->ndim,
        view->storage_offset,
        d_parent_shape,
        d_parent_strides,
        parent->ndim,
        parent->storage_offset,
        result->storage->data,
        d_result_strides,
        result->storage_offset,
        grad->numel
    );

    cudaDeviceSynchronize();

    cudaFree(d_grad_shape);
    cudaFree(d_grad_strides);
    cudaFree(d_view_shape);
    cudaFree(d_view_strides);
    cudaFree(d_parent_shape);
    cudaFree(d_parent_strides);
    cudaFree(d_result_strides);

    return result;
}