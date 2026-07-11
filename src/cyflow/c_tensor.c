#include "c_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Stride Calculation
// ============================================================================
void compute_contiguous_strides(const int64_t *shape, size_t ndim,
                                int64_t *strides_out) {
  if (ndim == 0)
    return;

  int64_t current_stride = 1;
  // Iterate backwards from the last dimension to compute C-contiguous strides
  for (int64_t i = (int64_t)ndim - 1; i >= 0; i--) {
    strides_out[i] = current_stride;
    current_stride *= shape[i];
  }
}

// ============================================================================
// Storage Allocation & Reference Counting
// ============================================================================

// Allocate zero-initialized memory buffer on the heap
Storage *storage_create(size_t size) {
  Storage *storage = (Storage *)malloc(sizeof(Storage));
  if (!storage)
    return NULL;

  // calloc zero-initializes the memory buffer
  storage->data = (float *)calloc(size, sizeof(float));
  if (!storage->data && size > 0) {
    free(storage);
    return NULL;
  }

  storage->size = size;
  storage->ref_count = 1; // Starts with 1 owner
  return storage;
}

// Increment reference count when a new Tensor view shares this Storage
void storage_retain(Storage *storage) {
  if (storage != NULL) {
    storage->ref_count++;
  }
}

// Decrement reference count; free underlying memory only when count reaches 0
void storage_free(Storage *storage) {
  if (!storage)
    return;

  storage->ref_count--;
  if (storage->ref_count == 0) {
    if (storage->data) {
      free(storage->data);
    }
    free(storage);
  }
}

// ============================================================================
// TensorImpl Operations
// ============================================================================

// Allocate a brand-new base Tensor and its underlying Storage
TensorImpl *tensor_create(const int64_t *shape, size_t ndim) {
  size_t numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    numel *= (size_t)shape[i];
  }

  // 1. Allocate storage buffer
  Storage *storage = storage_create(numel);
  if (!storage)
    return NULL;

  // 2. Allocate TensorImpl metadata
  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor) {
    storage_free(storage);
    return NULL;
  }

  tensor->storage = storage;
  tensor->ndim = ndim;
  tensor->numel = numel;
  tensor->storage_offset = 0;

  // 3. Allocate shape and stride arrays
  if (ndim > 0) {
    tensor->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
    tensor->strides = (int64_t *)malloc(ndim * sizeof(int64_t));

    if (!tensor->shape || !tensor->strides) {
      tensor_free(tensor);
      return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
    compute_contiguous_strides(shape, ndim, tensor->strides);
  } else {
    tensor->shape = NULL;
    tensor->strides = NULL;
  }

  return tensor;
}

// Create a non-owning view that shares existing Storage with custom metadata
TensorImpl *tensor_create_view(Storage *storage, const int64_t *shape,
                               const int64_t *strides, size_t ndim,
                               size_t storage_offset) {
  if (!storage)
    return NULL;

  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor)
    return NULL;

  size_t numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    numel *= (size_t)shape[i];
  }

  // Retain storage so it stays alive as long as this view exists
  storage_retain(storage);

  tensor->storage = storage;
  tensor->ndim = ndim;
  tensor->numel = numel;
  tensor->storage_offset = storage_offset;

  if (ndim > 0) {
    tensor->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
    tensor->strides = (int64_t *)malloc(ndim * sizeof(int64_t));

    if (!tensor->shape || !tensor->strides) {
      tensor_free(tensor);
      return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
    memcpy(tensor->strides, strides, ndim * sizeof(int64_t));
  } else {
    tensor->shape = NULL;
    tensor->strides = NULL;
  }

  return tensor;
}

// Free TensorImpl metadata and decrement reference count on Storage
void tensor_free(TensorImpl *tensor) {
  if (!tensor)
    return;

  if (tensor->storage) {
    storage_free(tensor->storage);
  }
  if (tensor->shape) {
    free(tensor->shape);
  }
  if (tensor->strides) {
    free(tensor->strides);
  }

  free(tensor);
}
