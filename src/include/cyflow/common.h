#ifndef TENSOR_COMMON_H
#define TENSOR_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIM(a, b) ((a) > (b) ? (a) : (b))

typedef enum { DEVICE_CPU = 0, DEVICE_CUDA = 1 } DeviceType;

typedef struct {
  float *data; // Pointer to data (can be CPU or GPU pointer)
  size_t size; // Number of elements
  int ref_count;
  bool owns_data;
  DeviceType device; // Tracks if data is on CPU or GPU
} Storage;

typedef struct {
  Storage *storage;
  int64_t *shape;   // ALWAYS on the CPU
  int64_t *strides; // ALWAYS on the CPU
  size_t ndim;
  size_t numel;
  size_t storage_offset;
} TensorImpl;

static inline void compute_contiguous_strides(int64_t *strides,
                                              const int64_t *shape,
                                              size_t ndim) {
  if (ndim == 0)
    return;
  strides[ndim - 1] = 1;
  for (int i = (int)ndim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
}

static inline TensorImpl *tensor_view(TensorImpl *src, const int64_t *shape,
                                      size_t ndim) {
  if (!src || !src->storage)
    return NULL;

  size_t new_numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    new_numel *= shape[i];
  }

  if (new_numel != src->numel) {
    fprintf(stderr,
            "Cyflow Error: Cannot reshape tensor of size %zu into shape with "
            "size %zu\n",
            src->numel, new_numel);
    return NULL;
  }

  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;

  tensor->ndim = ndim;
  tensor->numel = new_numel;
  tensor->storage_offset = src->storage_offset;

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

  // Increment shared storage reference count
  tensor->storage = src->storage;
  tensor->storage->ref_count++;

  return tensor;
}

#endif
