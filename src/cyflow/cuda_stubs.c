#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void cuda_not_available_error(const char *func_name) {
  fprintf(
      stderr,
      "Cyflow Error: %s called, but Cyflow was built without CUDA support.\n",
      func_name);
  exit(1);
}

/* ============================================================================
 * Core Tensor Memory & Lifecycle Stubs
 * ============================================================================ */

Storage *storage_create_cuda(size_t size) {
  cuda_not_available_error("storage_create_cuda");
  return NULL;
}

void storage_free_cuda(Storage *storage) {
  // Safe no-op for garbage collection on CPU builds
}

TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim) {
  cuda_not_available_error("tensor_create_cuda");
  return NULL;
}

void tensor_free_cuda(TensorImpl *tensor) {
  // Safe no-op for garbage collection on CPU builds
}

void tensor_fill_uniform_cuda(TensorImpl *tensor) {
  cuda_not_available_error("tensor_fill_uniform_cuda");
}

void cyflow_manual_seed_cuda(unsigned long long seed) {
  cuda_not_available_error("cyflow_manual_seed_cuda");
}

void tensor_set_data_cuda(TensorImpl *tensor, const float *data) {
  cuda_not_available_error("tensor_set_data_cuda");
}

bool tensor_is_contiguous_cuda(const TensorImpl *tensor) {
  cuda_not_available_error("tensor_is_contiguous_cuda");
  return false;
}

/* ============================================================================
 * Inplace Scalar Math Stubs
 * ============================================================================ */

void tensor_add_scalar_cuda(TensorImpl *tensor, float val) {
  cuda_not_available_error("tensor_add_scalar_cuda");
}

void tensor_sub_scalar_cuda(TensorImpl *tensor, float val) {
  cuda_not_available_error("tensor_sub_scalar_cuda");
}

void tensor_mul_scalar_cuda(TensorImpl *tensor, float val) {
  cuda_not_available_error("tensor_mul_scalar_cuda");
}

void tensor_div_scalar_cuda(TensorImpl *tensor, float val) {
  cuda_not_available_error("tensor_div_scalar_cuda");
}

/* ============================================================================
 * Inplace Tensor Math Stubs
 * ============================================================================ */

void tensor_add_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  cuda_not_available_error("tensor_add_tensor_cuda");
}

void tensor_sub_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  cuda_not_available_error("tensor_sub_tensor_cuda");
}

void tensor_mul_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  cuda_not_available_error("tensor_mul_tensor_cuda");
}

void tensor_div_tensor_cuda(TensorImpl *dst, const TensorImpl *src) {
  cuda_not_available_error("tensor_div_tensor_cuda");
}

/* ============================================================================
 * CUDA Runtime API Stubs
 * ============================================================================ */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorNoDevice 38

typedef enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  cuda_not_available_error("cudaMemcpy");
  return cudaErrorNoDevice;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  cuda_not_available_error("cudaMalloc");
  return cudaErrorNoDevice;
}

cudaError_t cudaFree(void *devPtr) {
  // Safe no-op to allow destructor cleanup without crashing
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  return cudaSuccess;
}