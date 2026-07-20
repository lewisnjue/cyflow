#include "c_tensor.h"
#include <Python.h>
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX_DIM(a, b) ((a) > (b) ? (a) : (b))
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
  storage->owns_data = true;
  storage->owner = NULL;
  return storage;
}

Storage *storage_create_from_buffer(float *data, size_t size, void *owner) {
  Storage *storage = (Storage *)malloc(sizeof(Storage));
  if (!storage)
    return NULL;

  storage->data = data;
  storage->size = size;
  storage->ref_count = 1;
  storage->owns_data = false;
  storage->owner = owner;
  if (owner) {
    Py_INCREF((PyObject *)owner);
  }
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
    if (storage->owns_data && storage->data) {
      free(storage->data);
    }
    if (storage->owner) {
      Py_DECREF((PyObject *)storage->owner);
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

TensorImpl *tensor_create_from_buffer(float *data,
                                     const int64_t *shape,
                                     const int64_t *strides,
                                     size_t ndim,
                                     void *owner) {
  size_t numel = 1;
  for (size_t i = 0; i < ndim; i++) {
    numel *= (size_t)shape[i];
  }

  Storage *storage = storage_create_from_buffer(data, numel, owner);
  if (!storage)
    return NULL;

  TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
  if (!tensor) {
    storage_free(storage);
    return NULL;
  }

  tensor->storage = storage;
  tensor->ndim = ndim;
  tensor->numel = numel;
  tensor->storage_offset = 0;

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
bool broadcast_shapes(const int64_t *shape_a, size_t ndim_a,
                      const int64_t *shape_b, size_t ndim_b,
                      int64_t **out_shape, size_t *out_ndim) {
  size_t max_ndim = MAX_DIM(ndim_a, ndim_b);
  *out_ndim = max_ndim;

  if (max_ndim == 0) {
    *out_shape = NULL;
    return true;
  }

  *out_shape = (int64_t *)malloc(max_ndim * sizeof(int64_t));
  if (!*out_shape)
    return false;

  // Iterate backwards right-to-left
  for (int i = 0; i < (int)max_ndim; i++) {
    int idx_a = (int)ndim_a - 1 - i;
    int idx_b = (int)ndim_b - 1 - i;
    int idx_out = (int)max_ndim - 1 - i;

    // If one shape is shorter, treat its missing leading dimensions as size 1
    int64_t dim_a = (idx_a >= 0) ? shape_a[idx_a] : 1;
    int64_t dim_b = (idx_b >= 0) ? shape_b[idx_b] : 1;

    if (dim_a == dim_b) {
      (*out_shape)[idx_out] = dim_a;
    } else if (dim_a == 1) {
      (*out_shape)[idx_out] = dim_b;
    } else if (dim_b == 1) {
      (*out_shape)[idx_out] = dim_a;
    } else {
      // Dimensions are fundamentally incompatible (e.g., 3 and 4)
      free(*out_shape);
      *out_shape = NULL;
      return false;
    }
  }
  return true;
}

bool compute_broadcast_strides(const int64_t *orig_shape,
                               const int64_t *orig_strides, size_t orig_ndim,
                               const int64_t *bcast_shape, size_t bcast_ndim,
                               int64_t **out_strides) {
  if (bcast_ndim == 0) {
    *out_strides = NULL;
    return true;
  }

  *out_strides = (int64_t *)malloc(bcast_ndim * sizeof(int64_t));
  if (!*out_strides)
    return false;

  // Iterate backwards right-to-left
  for (int i = 0; i < (int)bcast_ndim; i++) {
    int idx_orig = (int)orig_ndim - 1 - i;
    int idx_bcast = (int)bcast_ndim - 1 - i;

    if (idx_orig < 0) {
      // Dimension was prepended to the original shape.
      // We set stride to 0 so the pointer stays on the same memory block.
      (*out_strides)[idx_bcast] = 0;
    } else {
      int64_t dim_orig = orig_shape[idx_orig];
      int64_t dim_bcast = bcast_shape[idx_bcast];

      if (dim_orig == 1 && dim_bcast > 1) {
        // Dimension was expanded from 1 to N. Stride becomes 0.
        (*out_strides)[idx_bcast] = 0;
      } else {
        // Dimension matches, keep the original stride.
        (*out_strides)[idx_bcast] = orig_strides[idx_orig];
      }
    }
  }
  return true;
}
// ============================================================================
// N-Dimensional Iterator Utilities
// ============================================================================

void index_to_coords(size_t flat_idx, const int64_t *shape, size_t ndim,
                     int64_t *coords_out) {
  if (ndim == 0)
    return;

  size_t current_idx = flat_idx;
  // Iterate backwards right-to-left
  for (int i = (int)ndim - 1; i >= 0; i--) {
    // Cast to size_t to prevent signed/unsigned mismatch warnings
    coords_out[i] = (int64_t)(current_idx % (size_t)shape[i]);
    current_idx /= (size_t)shape[i];
  }
}

size_t coords_to_offset(const int64_t *coords, const int64_t *strides,
                        size_t ndim) {
  size_t offset = 0;
  for (size_t i = 0; i < ndim; i++) {
    // Multiply coordinate by stride and accumulate
    offset += (size_t)(coords[i] * strides[i]);
  }
  return offset;
}

// ============================================================================
// Math Operations (BLAS)
// ============================================================================

TensorImpl *tensor_matmul(TensorImpl *A, TensorImpl *B) {
  // 1. Matmul requires at least 2D tensors (Matrix x Matrix)
  // Note: 1D vector x 1D vector usually requires a separate dot product
  // function or reshaping to 2D in Python before calling this.
  if (A->ndim < 2 || B->ndim < 2) {
    return NULL;
  }

  // 2. Extract matrix dimensions (last two dimensions)
  int64_t M = A->shape[A->ndim - 2];
  int64_t K = A->shape[A->ndim - 1];
  int64_t K_b = B->shape[B->ndim - 2];
  int64_t N = B->shape[B->ndim - 1];

  if (K != K_b) {
    return NULL; // Inner dimensions must match!
  }

  // Ensure the innermost dimension is contiguous (stride == 1).
  // If it's not, we would need to pass CblasTrans to OpenBLAS, which we'll
  // skip for this initial implementation to keep things manageable.
  if (A->strides[A->ndim - 1] != 1 || B->strides[B->ndim - 1] != 1) {
    return NULL;
  }

  // 3. Extract batch dimensions (everything before the last two)
  size_t batch_ndim_a = A->ndim - 2;
  size_t batch_ndim_b = B->ndim - 2;

  int64_t *bcast_batch_shape = NULL;
  size_t bcast_batch_ndim = 0;

  // Broadcast batch shapes
  if (!broadcast_shapes(A->shape, batch_ndim_a, B->shape, batch_ndim_b,
                        &bcast_batch_shape, &bcast_batch_ndim)) {
    return NULL; // Batch shapes are incompatible
  }

  // Compute zero-strides for broadcasting
  int64_t *bcast_strides_a = NULL, *bcast_strides_b = NULL;
  compute_broadcast_strides(A->shape, A->strides, batch_ndim_a,
                            bcast_batch_shape, bcast_batch_ndim,
                            &bcast_strides_a);
  compute_broadcast_strides(B->shape, B->strides, batch_ndim_b,
                            bcast_batch_shape, bcast_batch_ndim,
                            &bcast_strides_b);

  // 4. Calculate total number of matrix multiplications to perform
  size_t total_batches = 1;
  for (size_t i = 0; i < bcast_batch_ndim; i++) {
    total_batches *= (size_t)bcast_batch_shape[i];
  }

  // 5. Create the output tensor C
  size_t out_ndim = bcast_batch_ndim + 2;
  int64_t *out_shape = (int64_t *)malloc(out_ndim * sizeof(int64_t));

  if (bcast_batch_ndim > 0) {
    memcpy(out_shape, bcast_batch_shape, bcast_batch_ndim * sizeof(int64_t));
  }
  out_shape[out_ndim - 2] = M;
  out_shape[out_ndim - 1] = N;

  TensorImpl *C = tensor_create(out_shape, out_ndim);
  free(out_shape);

  if (!C)
    goto cleanup;

  // 6. Set up BLAS variables
  // lda, ldb, ldc are the strides between rows.
  int lda = (int)A->strides[A->ndim - 2];
  int ldb = (int)B->strides[B->ndim - 2];
  int ldc = (int)C->strides[C->ndim - 2];

  int64_t *coords = NULL;
  if (bcast_batch_ndim > 0) {
    coords = (int64_t *)malloc(bcast_batch_ndim * sizeof(int64_t));
  }

  // 7. Execute the batched matrix multiplications
  for (size_t b = 0; b < total_batches; b++) {
    size_t offset_a = 0;
    size_t offset_b = 0;

    // Calculate ND coordinates and offsets for the current batch
    if (bcast_batch_ndim > 0) {
      index_to_coords(b, bcast_batch_shape, bcast_batch_ndim, coords);
      offset_a = coords_to_offset(coords, bcast_strides_a, bcast_batch_ndim);
      offset_b = coords_to_offset(coords, bcast_strides_b, bcast_batch_ndim);
    }

    // C is guaranteed fully contiguous, so its offset is simply batch_index *
    // (M * N)
    size_t offset_c = b * (size_t)(M * N);

    // Apply base offsets + batch offsets
    float *ptr_a = A->storage->data + A->storage_offset + offset_a;
    float *ptr_b = B->storage->data + B->storage_offset + offset_b;
    float *ptr_c = C->storage->data + C->storage_offset + offset_c;

    // Perform C = A @ B for this batch chunk
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N,
                (int)K, 1.0f, ptr_a, lda, ptr_b, ldb, 0.0f, ptr_c, ldc);
  }

cleanup:
  if (bcast_batch_shape)
    free(bcast_batch_shape);
  if (bcast_strides_a)
    free(bcast_strides_a);
  if (bcast_strides_b)
    free(bcast_strides_b);
  if (coords)
    free(coords);

  return C;
}

// ============================================================================
// Element-Wise Operations (with Broadcasting)
// ============================================================================

// Macro to generate robust element-wise binary operations
#define IMPLEMENT_BINARY_OP(name, op)                                          \
  TensorImpl *tensor_##name(TensorImpl *A, TensorImpl *B) {                    \
    int64_t *out_shape = NULL;                                                 \
    size_t out_ndim = 0;                                                       \
                                                                               \
    /* 1. Calculate Broadcasted Shape */                                       \
    if (!broadcast_shapes(A->shape, A->ndim, B->shape, B->ndim, &out_shape,    \
                          &out_ndim)) {                                        \
      return NULL;                                                             \
    }                                                                          \
                                                                               \
    /* 2. Calculate Zero-Strides for both A and B */                           \
    int64_t *strides_a = NULL, *strides_b = NULL;                              \
    if (!compute_broadcast_strides(A->shape, A->strides, A->ndim, out_shape,   \
                                   out_ndim, &strides_a)) {                    \
      free(out_shape);                                                         \
      return NULL;                                                             \
    }                                                                          \
    if (!compute_broadcast_strides(B->shape, B->strides, B->ndim, out_shape,   \
                                   out_ndim, &strides_b)) {                    \
      free(out_shape);                                                         \
      free(strides_a);                                                         \
      return NULL;                                                             \
    }                                                                          \
                                                                               \
    /* 3. Allocate Output Tensor */                                            \
    TensorImpl *C = tensor_create(out_shape, out_ndim);                        \
    free(out_shape);                                                           \
    if (!C) {                                                                  \
      free(strides_a);                                                         \
      free(strides_b);                                                         \
      return NULL;                                                             \
    }                                                                          \
                                                                               \
    /* 4. N-Dimensional Iterator Loop */                                       \
    int64_t *coords = NULL;                                                    \
    if (out_ndim > 0) {                                                        \
      coords = (int64_t *)malloc(out_ndim * sizeof(int64_t));                  \
    }                                                                          \
                                                                               \
    for (size_t i = 0; i < C->numel; i++) {                                    \
      size_t off_a = 0, off_b = 0;                                             \
      if (out_ndim > 0) {                                                      \
        index_to_coords(i, C->shape, out_ndim, coords);                        \
        off_a = coords_to_offset(coords, strides_a, out_ndim);                 \
        off_b = coords_to_offset(coords, strides_b, out_ndim);                 \
      }                                                                        \
                                                                               \
      float val_a = A->storage->data[A->storage_offset + off_a];               \
      float val_b = B->storage->data[B->storage_offset + off_b];               \
                                                                               \
      /* Apply the math operation */                                           \
      C->storage->data[C->storage_offset + i] = (val_a op val_b);              \
    }                                                                          \
                                                                               \
    /* Cleanup */                                                              \
    if (strides_a) free(strides_a);                                            \
    if (strides_b) free(strides_b);                                            \
    if (coords) free(coords);                                                  \
                                                                               \
    return C;                                                                  \
  }

// Generate the specific functions
IMPLEMENT_BINARY_OP(add, +)
IMPLEMENT_BINARY_OP(sub, -)
IMPLEMENT_BINARY_OP(mul, *)

// ============================================================================
// Unary / Scalar Operations
// ============================================================================

TensorImpl *tensor_exp(TensorImpl *A) {
  TensorImpl *C = tensor_create(A->shape, A->ndim);
  if (!C) return NULL;

  int64_t *coords = NULL;
  if (A->ndim > 0) {
    coords = (int64_t *)malloc(A->ndim * sizeof(int64_t));
  }

  for (size_t i = 0; i < C->numel; i++) {
    size_t off_a = 0;
    if (A->ndim > 0) {
      index_to_coords(i, C->shape, A->ndim, coords);
      off_a = coords_to_offset(coords, A->strides, A->ndim);
    }

    float val = A->storage->data[A->storage_offset + off_a];
    C->storage->data[C->storage_offset + i] = expf(val);
  }

  if (coords) free(coords);
  return C;
}

TensorImpl *tensor_pow(TensorImpl *A, int64_t exponent) {
  TensorImpl *C = tensor_create(A->shape, A->ndim);
  if (!C) return NULL;

  int64_t *coords = NULL;
  if (A->ndim > 0) {
    coords = (int64_t *)malloc(A->ndim * sizeof(int64_t));
  }

  for (size_t i = 0; i < C->numel; i++) {
    size_t off_a = 0;
    if (A->ndim > 0) {
      index_to_coords(i, C->shape, A->ndim, coords);
      off_a = coords_to_offset(coords, A->strides, A->ndim);
    }

    float val = A->storage->data[A->storage_offset + off_a];
    // powf takes floats, so we cast the int64_t exponent
    C->storage->data[C->storage_offset + i] = powf(val, (float)exponent);
  }

  if (coords) free(coords);
  return C;
}