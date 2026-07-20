#ifndef C_TENSOR_H
#define C_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
// ============================================================================
// 1. Storage: Holds the actual underlying contiguous memory block
// ============================================================================
typedef struct {
  float *data;      // Raw float array allocated on the heap or borrowed from an external buffer
  size_t size;      // Total number of float elements allocated or referenced
  size_t ref_count; // Keeps track of how many TensorImpls point here
  bool owns_data;   // Whether we should free the underlying data buffer
  void *owner;      // Optional owner pointer for external memory sources
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
Storage *storage_create_from_buffer(float *data, size_t size, void *owner);
void storage_retain(Storage *storage);
void storage_free(Storage *storage);

// ============================================================================
// Function Declarations: Tensor Operations
// ============================================================================
// Creates a contiguous tensor from given shape and rank
TensorImpl *tensor_create(const int64_t *shape, size_t ndim);
TensorImpl *tensor_create_from_buffer(float *data,
                                     const int64_t *shape,
                                     const int64_t *strides,
                                     size_t ndim,
                                     void *owner);

// Creates a slice view sharing the same storage
TensorImpl *tensor_create_view(Storage *storage, const int64_t *shape,
                               const int64_t *strides, size_t ndim,
                               size_t storage_offset);

void tensor_free(TensorImpl *tensor);

// Utility: Calculate row-major contiguous strides for a given shape
void compute_contiguous_strides(const int64_t *shape, size_t ndim,
                                int64_t *strides_out);

// ============================================================================
// Broadcasting Utilities
// ============================================================================

// Compares two shapes right-to-left and allocates the broadcasted out_shape.
// Returns false if shapes are incompatible.
bool broadcast_shapes(const int64_t *shape_a, size_t ndim_a,
                      const int64_t *shape_b, size_t ndim_b,
                      int64_t **out_shape, size_t *out_ndim);

// Maps an original tensor's strides to a new broadcasted shape.
// Any dimension expanded from 1 to N, or prepended, gets a stride of 0.
bool compute_broadcast_strides(const int64_t *orig_shape,
                               const int64_t *orig_strides, size_t orig_ndim,
                               const int64_t *bcast_shape, size_t bcast_ndim,
                               int64_t **out_strides);

// ============================================================================
// N-Dimensional Iterator Utilities
// ============================================================================

// Converts a flat 1D index into N-dimensional coordinates based on the shape.
// coords_out must be pre-allocated to size `ndim`.
void index_to_coords(size_t flat_idx, const int64_t *shape, size_t ndim,
                     int64_t *coords_out);

// Converts N-dimensional coordinates into a flat memory offset using strides.
size_t coords_to_offset(const int64_t *coords, const int64_t *strides,
                        size_t ndim);
// ============================================================================
// Math Operations
// ============================================================================

TensorImpl *tensor_matmul(TensorImpl *A, TensorImpl *B);
TensorImpl *tensor_add(TensorImpl *A, TensorImpl *B);
TensorImpl *tensor_sub(TensorImpl *A, TensorImpl *B);
TensorImpl *tensor_mul(TensorImpl *A, TensorImpl *B);
TensorImpl *tensor_pow(TensorImpl *A, int64_t exponent);
TensorImpl *tensor_exp(TensorImpl *A);
#endif // C_TENSOR_H
