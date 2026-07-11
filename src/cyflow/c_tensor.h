#ifndef C_TENSOR_H
#define C_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// ============================================================================
// 1. Storage: Holds the actual underlying contiguous memory block
// ============================================================================
typedef struct {
  float *data;      // Raw float array allocated on the heap
  size_t size;      // Total number of float elements allocated
  size_t ref_count; // Keeps track of how many TensorImpls point here
} Storage;

// ============================================================================
// 2. TensorImpl: Metadata describing how to view and navigate the Storage
// ============================================================================
typedef struct {
  Storage *storage; // Pointer to shared heap memory storage
  int64_t *shape;   // Array of size `ndim` representing dimension sizes
  int64_t *strides; // Array of size `ndim` representing memory jump steps
  size_t ndim;      // Number of dimensions (Rank)
  size_t
      storage_offset; // Index offset into storage->data where this view starts
  size_t numel; // Total number of elements in this view (product of shape)
} TensorImpl;

// ============================================================================
// Function Declarations: Storage Operations
// ============================================================================
Storage *storage_create(size_t size);
void storage_retain(Storage *storage);
void storage_free(Storage *storage);

// ============================================================================
// Function Declarations: Tensor Operations
// ============================================================================
// Creates a contiguous tensor from given shape and rank
TensorImpl *tensor_create(const int64_t *shape, size_t ndim);

// Creates a slice view sharing the same storage
TensorImpl *tensor_create_view(Storage *storage, const int64_t *shape,
                               const int64_t *strides, size_t ndim,
                               size_t storage_offset);

void tensor_free(TensorImpl *tensor);

// Utility: Calculate row-major contiguous strides for a given shape
void compute_contiguous_strides(const int64_t *shape, size_t ndim,
                                int64_t *strides_out);

#endif // C_TENSOR_H
