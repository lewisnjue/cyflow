#include "cyflow/common.h"
#include "cyflow/inline_op_cuda.h"
#include "cyflow/tensor.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>

__global__ void kernel_add_scalar_contiguous(float *data, float val,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    data[i] += val;
  }
}

__global__ void kernel_mul_scalar_contiguous(float *data, float val,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    data[i] *= val;
  }
}

__global__ void kernel_add_scalar_strided(float *data, float val, size_t numel,
                                          TensorMeta meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    data[get_physical_offset(i, meta)] += val;
  }
}

__global__ void kernel_mul_scalar_strided(float *data, float val, size_t numel,
                                          TensorMeta meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    data[get_physical_offset(i, meta)] *= val;
  }
}
__device__ inline size_t get_physical_offset(size_t linear_idx,
                                             TensorMeta meta) {
  size_t offset = meta.storage_offset;
  for (int k = (int)meta.ndim - 1; k >= 0; --k) {
    size_t dim_idx = linear_idx % meta.shape[k];
    offset += dim_idx * meta.strides[k];
    linear_idx /= meta.shape[k];
  }
  return offset;
}

__global__ void kernel_add_tensor_contiguous(float *dst, const float *src,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride)
    dst[i] += src[i];
}

__global__ void kernel_sub_tensor_contiguous(float *dst, const float *src,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride)
    dst[i] -= src[i];
}

__global__ void kernel_mul_tensor_contiguous(float *dst, const float *src,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride)
    dst[i] *= src[i];
}

__global__ void kernel_div_tensor_contiguous(float *dst, const float *src,
                                             size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride)
    dst[i] /= src[i];
}

__global__ void kernel_add_tensor_strided(float *dst, const float *src,
                                          size_t numel, TensorMeta dst_meta,
                                          TensorMeta src_meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    dst[get_physical_offset(i, dst_meta)] +=
        src[get_physical_offset(i, src_meta)];
  }
}

__global__ void kernel_sub_tensor_strided(float *dst, const float *src,
                                          size_t numel, TensorMeta dst_meta,
                                          TensorMeta src_meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    dst[get_physical_offset(i, dst_meta)] -=
        src[get_physical_offset(i, src_meta)];
  }
}

__global__ void kernel_mul_tensor_strided(float *dst, const float *src,
                                          size_t numel, TensorMeta dst_meta,
                                          TensorMeta src_meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    dst[get_physical_offset(i, dst_meta)] *=
        src[get_physical_offset(i, src_meta)];
  }
}

__global__ void kernel_div_tensor_strided(float *dst, const float *src,
                                          size_t numel, TensorMeta dst_meta,
                                          TensorMeta src_meta) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    dst[get_physical_offset(i, dst_meta)] /=
        src[get_physical_offset(i, src_meta)];
  }
}

extern "C" {

void tensor_add_scalar_cuda(TensorImpl *dst, float val) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst)) {
    float *d_ptr = dst->storage->data + dst->storage_offset;
    kernel_add_scalar_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_ptr, val,
                                                                     numel);
  } else {
    TensorMeta meta = create_tensor_meta(dst);
    kernel_add_scalar_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, val, numel, meta);
  }
}

void tensor_sub_scalar_cuda(TensorImpl *dst, float val) {
  tensor_add_scalar_cuda(dst, -val);
}

void tensor_mul_scalar_cuda(TensorImpl *dst, float val) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst)) {
    float *d_ptr = dst->storage->data + dst->storage_offset;
    kernel_mul_scalar_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_ptr, val,
                                                                     numel);
  } else {
    TensorMeta meta = create_tensor_meta(dst);
    kernel_mul_scalar_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, val, numel, meta);
  }
}

void tensor_div_scalar_cuda(TensorImpl *dst, float val) {
  tensor_mul_scalar_cuda(dst, 1.0f / val);
}

void tensor_add_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    float *d_dst = dst->storage->data + dst->storage_offset;
    const float *d_src = src->storage->data + src->storage_offset;
    kernel_add_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        d_dst, d_src, numel);
  } else {
    TensorMeta dst_meta = create_tensor_meta(dst);
    TensorMeta src_meta = create_tensor_meta(src);
    kernel_add_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, src->storage->data, numel, dst_meta, src_meta);
  }
}

void tensor_sub_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    float *d_dst = dst->storage->data + dst->storage_offset;
    const float *d_src = src->storage->data + src->storage_offset;
    kernel_sub_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        d_dst, d_src, numel);
  } else {
    TensorMeta dst_meta = create_tensor_meta(dst);
    TensorMeta src_meta = create_tensor_meta(src);
    kernel_sub_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, src->storage->data, numel, dst_meta, src_meta);
  }
}

void tensor_mul_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    float *d_dst = dst->storage->data + dst->storage_offset;
    const float *d_src = src->storage->data + src->storage_offset;
    kernel_mul_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        d_dst, d_src, numel);
  } else {
    TensorMeta dst_meta = create_tensor_meta(dst);
    TensorMeta src_meta = create_tensor_meta(src);
    kernel_mul_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, src->storage->data, numel, dst_meta, src_meta);
  }
}

void tensor_div_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  size_t numel = dst->numel;
  if (numel == 0)
    return;
  int blocks = (numel + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
    float *d_dst = dst->storage->data + dst->storage_offset;
    const float *d_src = src->storage->data + src->storage_offset;
    kernel_div_tensor_contiguous<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        d_dst, d_src, numel);
  } else {
    TensorMeta dst_meta = create_tensor_meta(dst);
    TensorMeta src_meta = create_tensor_meta(src);
    kernel_div_tensor_strided<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
        dst->storage->data, src->storage->data, numel, dst_meta, src_meta);
  }
}

} // extern "C"
