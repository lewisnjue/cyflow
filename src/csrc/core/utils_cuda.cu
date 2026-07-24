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
    
    size_t ndim_diff = grad->ndim - target_ndim;
    
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