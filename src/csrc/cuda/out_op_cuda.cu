#include "cyflow/out_op_cpu.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define MAX_CUDA_DIMS 8 // Support up to 8D tensors without dynamic memory allocation

// Struct passed by value to kernels to avoid cudaMalloc for shapes/strides
struct CudaDimInfo {
    size_t ndim;
    int64_t shape[MAX_CUDA_DIMS];
    int64_t strides1[MAX_CUDA_DIMS];
    int64_t strides2[MAX_CUDA_DIMS];
};

// ==========================================
// KERNEL DEFINITIONS
// ==========================================

// 1. Contiguous Scalar Kernel
#define DEFINE_SCALAR_CONTIGUOUS_KERNEL(name, op) \
__global__ void name##_kernel(float *dst, const float *src, float val, size_t numel) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < numel) { \
        dst[idx] = src[idx] op val; \
    } \
}

// 2. Strided Scalar Kernel
#define DEFINE_SCALAR_STRIDED_KERNEL(name, op) \
__global__ void name##_kernel(float *dst, const float *src, float val, size_t numel, CudaDimInfo info) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < numel) { \
        size_t curr = idx; \
        size_t phys_offset = 0; \
        for (int k = (int)info.ndim - 1; k >= 0; --k) { \
            size_t coord = curr % info.shape[k]; \
            phys_offset += coord * info.strides1[k]; \
            curr /= info.shape[k]; \
        } \
        dst[idx] = src[phys_offset] op val; \
    } \
}

// 3. Contiguous Tensor Kernel
#define DEFINE_TENSOR_CONTIGUOUS_KERNEL(name, op) \
__global__ void name##_kernel(float *dst, const float *src1, const float *src2, size_t numel) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < numel) { \
        dst[idx] = src1[idx] op src2[idx]; \
    } \
}

// 4. Strided Tensor Kernel (Handles Broadcasting)
#define DEFINE_TENSOR_STRIDED_KERNEL(name, op) \
__global__ void name##_kernel(float *dst, const float *src1, const float *src2, size_t numel, CudaDimInfo info) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < numel) { \
        size_t curr = idx; \
        size_t off1 = 0, off2 = 0; \
        for (int k = (int)info.ndim - 1; k >= 0; --k) { \
            size_t coord = curr % info.shape[k]; \
            off1 += coord * info.strides1[k]; \
            off2 += coord * info.strides2[k]; \
            curr /= info.shape[k]; \
        } \
        dst[idx] = src1[off1] op src2[off2]; \
    } \
}

// Stamp out the kernels using the macros
DEFINE_SCALAR_CONTIGUOUS_KERNEL(add_scalar_contig, +)
DEFINE_SCALAR_CONTIGUOUS_KERNEL(sub_scalar_contig, -)
DEFINE_SCALAR_CONTIGUOUS_KERNEL(mul_scalar_contig, *)
DEFINE_SCALAR_CONTIGUOUS_KERNEL(div_scalar_contig, /)

DEFINE_SCALAR_STRIDED_KERNEL(add_scalar_strided, +)
DEFINE_SCALAR_STRIDED_KERNEL(sub_scalar_strided, -)
DEFINE_SCALAR_STRIDED_KERNEL(mul_scalar_strided, *)
DEFINE_SCALAR_STRIDED_KERNEL(div_scalar_strided, /)

DEFINE_TENSOR_CONTIGUOUS_KERNEL(add_tensor_contig, +)
DEFINE_TENSOR_CONTIGUOUS_KERNEL(sub_tensor_contig, -)
DEFINE_TENSOR_CONTIGUOUS_KERNEL(mul_tensor_contig, *)
DEFINE_TENSOR_CONTIGUOUS_KERNEL(div_tensor_contig, /)

DEFINE_TENSOR_STRIDED_KERNEL(add_tensor_strided, +)
DEFINE_TENSOR_STRIDED_KERNEL(sub_tensor_strided, -)
DEFINE_TENSOR_STRIDED_KERNEL(mul_tensor_strided, *)
DEFINE_TENSOR_STRIDED_KERNEL(div_tensor_strided, /)

// ==========================================
// HOST WRAPPER MACROS (The C API)
// ==========================================

#define CUDA_LAUNCH_SCALAR_CONTIG(name, dst, src, val) \
    size_t numel = dst->numel; \
    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; \
    float *d_ptr = dst->storage->data + dst->storage_offset; \
    const float *s_ptr = src->storage->data + src->storage_offset; \
    name##_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ptr, s_ptr, val, numel);

#define CUDA_LAUNCH_SCALAR_STRIDED(name, dst, src, val) \
    size_t numel = dst->numel; \
    if (dst->ndim > MAX_CUDA_DIMS) { printf("Error: Max %d dims supported\n", MAX_CUDA_DIMS); return; } \
    CudaDimInfo info; \
    info.ndim = dst->ndim; \
    for(int i=0; i<dst->ndim; i++) { \
        info.shape[i] = dst->shape[i]; \
        info.strides1[i] = src->strides[i]; \
    } \
    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; \
    float *d_ptr = dst->storage->data + dst->storage_offset; \
    const float *s_ptr = src->storage->data + src->storage_offset; \
    name##_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ptr, s_ptr, val, numel, info);

#define CUDA_LAUNCH_TENSOR_CONTIG(name, dst, src1, src2) \
    size_t numel = dst->numel; \
    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; \
    float *d_ptr = dst->storage->data + dst->storage_offset; \
    const float *s1_ptr = src1->storage->data + src1->storage_offset; \
    const float *s2_ptr = src2->storage->data + src2->storage_offset; \
    name##_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ptr, s1_ptr, s2_ptr, numel);

#define CUDA_LAUNCH_TENSOR_STRIDED(name, dst, src1, src2) \
    size_t numel = dst->numel; \
    if (dst->ndim > MAX_CUDA_DIMS) { printf("Error: Max %d dims supported\n", MAX_CUDA_DIMS); return; } \
    CudaDimInfo info; \
    info.ndim = dst->ndim; \
    for (int d = (int)info.ndim - 1; d >= 0; --d) { \
        info.shape[d] = dst->shape[d]; \
        int s1_idx = d - (info.ndim - src1->ndim); \
        info.strides1[d] = (s1_idx >= 0 && src1->shape[s1_idx] != 1) ? src1->strides[s1_idx] : 0; \
        int s2_idx = d - (info.ndim - src2->ndim); \
        info.strides2[d] = (s2_idx >= 0 && src2->shape[s2_idx] != 1) ? src2->strides[s2_idx] : 0; \
    } \
    int blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; \
    float *d_ptr = dst->storage->data + dst->storage_offset; \
    const float *s1_ptr = src1->storage->data + src1->storage_offset; \
    const float *s2_ptr = src2->storage->data + src2->storage_offset; \
    name##_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_ptr, s1_ptr, s2_ptr, numel, info);


// ==========================================
// C API IMPLEMENTATIONS
// ==========================================

extern "C" {

// Scalar Contiguous
void tensor_add_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_CONTIG(add_scalar_contig, dst, src, val) }
void tensor_sub_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_CONTIG(sub_scalar_contig, dst, src, val) }
void tensor_mul_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_CONTIG(mul_scalar_contig, dst, src, val) }
void tensor_div_out_scalar_contiguous_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_CONTIG(div_scalar_contig, dst, src, val) }

// Scalar Strided
void tensor_add_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_STRIDED(add_scalar_strided, dst, src, val) }
void tensor_sub_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_STRIDED(sub_scalar_strided, dst, src, val) }
void tensor_mul_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_STRIDED(mul_scalar_strided, dst, src, val) }
void tensor_div_out_scalar_strided_cuda(TensorImpl *dst, const TensorImpl *src, float val) { CUDA_LAUNCH_SCALAR_STRIDED(div_scalar_strided, dst, src, val) }

// Tensor Contiguous
void tensor_add_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_CONTIG(add_tensor_contig, dst, src1, src2) }
void tensor_sub_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_CONTIG(sub_tensor_contig, dst, src1, src2) }
void tensor_mul_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_CONTIG(mul_tensor_contig, dst, src1, src2) }
void tensor_div_out_tensor_contiguous_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_CONTIG(div_tensor_contig, dst, src1, src2) }

// Tensor Strided
void tensor_add_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_STRIDED(add_tensor_strided, dst, src1, src2) }
void tensor_sub_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_STRIDED(sub_tensor_strided, dst, src1, src2) }
void tensor_mul_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_STRIDED(mul_tensor_strided, dst, src1, src2) }
void tensor_div_out_tensor_strided_cuda(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) { CUDA_LAUNCH_TENSOR_STRIDED(div_tensor_strided, dst, src1, src2) }

} // extern "C"