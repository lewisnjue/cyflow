#include "cyflow/matmul_cuda.h"
#include "cyflow/tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define MATMUL_MAX_DIMS 8

struct MatmulBatchInfo {
    size_t batch_ndim;
    int64_t batch_shape[MATMUL_MAX_DIMS];
    int64_t a_batch_strides[MATMUL_MAX_DIMS];
    int64_t b_batch_strides[MATMUL_MAX_DIMS];
    int64_t d_batch_strides[MATMUL_MAX_DIMS];
    int64_t a_offset, b_offset, d_offset;
    int64_t a_row_stride, a_col_stride;
    int64_t b_row_stride, b_col_stride;
    int64_t d_row_stride, d_col_stride;
    int64_t M, K, N;
};

__global__ void matmul_kernel(float *dst, const float *a, const float *b,
                               MatmulBatchInfo info, size_t batch_numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_numel * (size_t)info.M * (size_t)info.N;
    if (idx >= total) return;

    size_t n = idx % (size_t)info.N;
    size_t tmp = idx / (size_t)info.N;
    size_t m = tmp % (size_t)info.M;
    size_t bi = tmp / (size_t)info.M;

    int64_t a_off = info.a_offset;
    int64_t b_off = info.b_offset;
    int64_t d_off = info.d_offset;

    size_t temp = bi;
    for (int d = (int)info.batch_ndim - 1; d >= 0; d--) {
        int64_t dim_size = info.batch_shape[d];
        int64_t coord = (int64_t)(temp % (size_t)dim_size);
        temp /= (size_t)dim_size;
        a_off += coord * info.a_batch_strides[d];
        b_off += coord * info.b_batch_strides[d];
        d_off += coord * info.d_batch_strides[d];
    }

    int64_t a_base = a_off + (int64_t)m * info.a_row_stride;
    int64_t b_base = b_off + (int64_t)n * info.b_col_stride;

    float sum = 0.0f;
    for (int64_t k = 0; k < info.K; k++) {
        sum += a[a_base + k * info.a_col_stride] * b[b_base + k * info.b_row_stride];
    }
    dst[d_off + (int64_t)m * info.d_row_stride + (int64_t)n * info.d_col_stride] = sum;
}

extern "C" void tensor_matmul_out_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
    size_t ndim1 = src1->ndim, ndim2 = src2->ndim, out_ndim = dst->ndim;
    size_t batch_ndim1 = ndim1 - 2, batch_ndim2 = ndim2 - 2;
    size_t out_batch_ndim = out_ndim - 2;

    if (out_batch_ndim > MATMUL_MAX_DIMS) {
        fprintf(stderr, "Cyflow Error: matmul batch dims exceed max supported (%d)\n", MATMUL_MAX_DIMS);
        return;
    }

    MatmulBatchInfo info;
    info.batch_ndim = out_batch_ndim;
    info.M = src1->shape[ndim1 - 2];
    info.K = src1->shape[ndim1 - 1];
    info.N = src2->shape[ndim2 - 1];
    info.a_row_stride = src1->strides[ndim1 - 2];
    info.a_col_stride = src1->strides[ndim1 - 1];
    info.b_row_stride = src2->strides[ndim2 - 2];
    info.b_col_stride = src2->strides[ndim2 - 1];
    info.d_row_stride = dst->strides[out_ndim - 2];
    info.d_col_stride = dst->strides[out_ndim - 1];
    info.a_offset = (int64_t)src1->storage_offset;
    info.b_offset = (int64_t)src2->storage_offset;
    info.d_offset = (int64_t)dst->storage_offset;

    size_t batch_numel = 1;
    for (size_t d = 0; d < out_batch_ndim; d++) {
        info.batch_shape[d] = dst->shape[d];
        batch_numel *= (size_t)dst->shape[d];

        int64_t a_idx = (int64_t)d - (int64_t)(out_batch_ndim - batch_ndim1);
        info.a_batch_strides[d] =
            (a_idx >= 0 && src1->shape[a_idx] != 1) ? src1->strides[a_idx] : 0;

        int64_t b_idx = (int64_t)d - (int64_t)(out_batch_ndim - batch_ndim2);
        info.b_batch_strides[d] =
            (b_idx >= 0 && src2->shape[b_idx] != 1) ? src2->strides[b_idx] : 0;

        info.d_batch_strides[d] = dst->strides[d];
    }

    size_t total = batch_numel * (size_t)info.M * (size_t)info.N;
    if (total == 0) return;

    int blocks = (int)((total + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
    matmul_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, src1->storage->data, src2->storage->data, info, batch_numel);
}
