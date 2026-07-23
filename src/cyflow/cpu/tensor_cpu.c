#include "cyflow/tensor_cpu.h"
#include "cyflow/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Storage *storage_create_cpu(size_t size) {
  Storage *storage = (Storage *)malloc(sizeof(Storage));
  if (!storage)
    return NULL;

  // calloc zero-initializes the memory buffer on the host CPU
  storage->data = (float *)calloc(size, sizeof(float));
  if (!storage->data && size > 0) {
    free(storage);
    return NULL;
  }

  storage->size = size;
  storage->ref_count = 1;
  storage->owns_data = true;
  storage->device = DEVICE_CPU; // Explicitly tag as CPU memory
  return storage;
}

void storage_free_cpu(Storage *storage) {
  if (!storage)
    return;

  storage->ref_count--;
  if (storage->ref_count == 0) {
    if (storage->owns_data && storage->data) {
      free(storage->data); // Standard CPU free
    }
    free(storage);
  }
}
TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim) {
  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;

  tensor->ndim = ndim;
  tensor->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
  tensor->strides = (int64_t *)malloc(ndim * sizeof(int64_t));
  if (!tensor->shape || !tensor->strides) {
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return NULL;
  }
  memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
  compute_contiguous_strides(tensor->strides, tensor->shape, ndim);
  size_t total_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }
  tensor->numel = total_size;
  tensor->storage_offset = 0;

  tensor->storage = storage_create_cpu(total_size);
  if (!tensor->storage) {
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return NULL;
  }
  return tensor;
}

void tensor_free_cpu(TensorImpl *tensor) {
  if (!tensor)
    return;
  if (tensor->storage) {
    storage_free_cpu(tensor->storage);
  }
  if (tensor->shape)
    free(tensor->shape);
  if (tensor->strides)
    free(tensor->strides);
  free(tensor);
}

void tensor_fill_uniform_cpu(TensorImpl *tensor) { // i will improve later
  if (!tensor || !tensor->storage || !tensor->storage->data)
    return;
  float *data = tensor->storage->data;
  size_t numel = tensor->numel;

  for (size_t i = 0; i < numel; i++) {
    data[i] = (float)rand() / (float)RAND_MAX;
  }
}
void cyflow_manual_seed(unsigned int seed) { srand(seed); }

void tensor_set_data_cpu(TensorImpl *tensor, const float *data) {
  if (!tensor || !tensor->storage || !tensor->storage->data || !data)
    return;
  memcpy(tensor->storage->data, data, tensor->numel * sizeof(float));
}
