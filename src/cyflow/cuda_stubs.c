#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"
#include <stdio.h>
#include <stdlib.h>

static void cuda_not_available_error(const char *func_name) {
  fprintf(
      stderr,
      "Cyflow Error: %s called, but Cyflow was built without CUDA support.\n",
      func_name);
  exit(1);
}

Storage *storage_create_cuda(size_t size) {
  cuda_not_available_error("storage_create_cuda");
  return NULL;
}

void storage_free_cuda(Storage *storage) {
  // No-op for stubs
}

TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim) {
  cuda_not_available_error("tensor_create_cuda");
  return NULL;
}

void tensor_free_cuda(TensorImpl *tensor) {
  // No-op for stubs
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
