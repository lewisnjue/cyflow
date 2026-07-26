#include "cyflow/utils.h"
#include "cyflow/tensor.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int compute_broadcast_shape(const int64_t *shape_a, size_t ndim_a,
                            const int64_t *shape_b, size_t ndim_b,
                            int64_t *out_shape, size_t *out_ndim) {
  size_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
  *out_ndim = max_ndim;

  for (size_t i = 1; i <= max_ndim; i++) {
    int64_t dim_a = (i <= ndim_a) ? shape_a[ndim_a - i] : 1;
    int64_t dim_b = (i <= ndim_b) ? shape_b[ndim_b - i] : 1;

    if (dim_a == dim_b) {
      out_shape[max_ndim - i] = dim_a;
    } else if (dim_a == 1) {
      out_shape[max_ndim - i] = dim_b;
    } else if (dim_b == 1) {
      out_shape[max_ndim - i] = dim_a;
    } else {
      return -1;
    }
  }
  return 0;
}

TensorImpl *tensor_unbroadcast_cpu(const TensorImpl *grad,
                                   const int64_t *target_shape,
                                   size_t target_ndim) {
  TensorImpl *result = tensor_create_cpu(target_shape, target_ndim);
  if (!result)
    return NULL;

  memset(result->storage->data, 0, result->storage->size * sizeof(float));
  if (grad->numel == 0)
    return result;

  int64_t *grad_coords = (int64_t *)malloc(grad->ndim * sizeof(int64_t));
  int64_t *target_coords = (int64_t *)malloc(target_ndim * sizeof(int64_t));
  size_t ndim_diff =
      (grad->ndim >= target_ndim) ? (grad->ndim - target_ndim) : 0;

  for (size_t i = 0; i < grad->numel; i++) {
    size_t temp = i;

    for (ptrdiff_t d = grad->ndim - 1; d >= 0; d--) {
      grad_coords[d] = temp % grad->shape[d];
      temp /= grad->shape[d];
    }

    for (size_t d = 0; d < target_ndim; d++) {
      if (target_shape[d] == 1) {
        target_coords[d] = 0;
      } else {
        target_coords[d] = grad_coords[ndim_diff + d];
      }
    }

    size_t grad_offset = grad->storage_offset;
    for (size_t d = 0; d < grad->ndim; d++) {
      grad_offset += grad_coords[d] * grad->strides[d];
    }

    size_t target_offset = result->storage_offset;
    for (size_t d = 0; d < target_ndim; d++) {
      target_offset += target_coords[d] * result->strides[d];
    }

    result->storage->data[target_offset] += grad->storage->data[grad_offset];
  }

  free(grad_coords);
  free(target_coords);

  return result;
}

static float read_tensor_element(const TensorImpl *tensor,
                                 size_t linear_index) {
  size_t temp = linear_index;
  size_t offset = tensor->storage_offset;

  for (ptrdiff_t d = (ptrdiff_t)tensor->ndim - 1; d >= 0; d--) {
    int64_t coord = temp % tensor->shape[d];
    temp /= tensor->shape[d];
    offset += (size_t)(coord * tensor->strides[d]);
  }

  return tensor->storage->data[offset];
}
static void offset_to_parent_coords(const TensorImpl *parent,
                                    size_t storage_offset,
                                    int64_t *parent_coords) {
  size_t rel = storage_offset - parent->storage_offset;
  size_t remaining = rel;

  // Track which dimensions we've already processed
  int *used = (int *)calloc(parent->ndim, sizeof(int));

  // Decode from largest stride to smallest stride, regardless of axis order
  for (size_t step = 0; step < parent->ndim; step++) {
    int64_t max_stride = -1;
    int best_d = -1;

    for (size_t d = 0; d < parent->ndim; d++) {
      if (!used[d] && parent->strides[d] > max_stride) {
        max_stride = parent->strides[d];
        best_d = (int)d;
      }
    }

    if (best_d == -1)
      break;
    used[best_d] = 1;

    if (parent->shape[best_d] == 0 || max_stride == 0) {
      parent_coords[best_d] = 0;
      continue;
    }

    parent_coords[best_d] = (int64_t)(remaining / (size_t)max_stride);
    remaining %= (size_t)max_stride;
  }

  free(used);
}
TensorImpl *tensor_accumulate_view_grad_cpu(const TensorImpl *parent,
                                            const TensorImpl *view,
                                            const TensorImpl *grad) {
  TensorImpl *result = NULL;
  int64_t *parent_coords = NULL;
  size_t view_offset = 0;
  size_t result_offset = 0;
  size_t i = 0;
  size_t d = 0;
  float grad_val = 0.0f;

  if (!parent || !view || !grad) {
    return NULL;
  }

  result = tensor_create_cpu(parent->shape, parent->ndim);
  if (!result) {
    return NULL;
  }

  memset(result->storage->data, 0, result->storage->size * sizeof(float));
  if (grad->numel == 0) {
    return result;
  }

  parent_coords = (int64_t *)malloc(parent->ndim * sizeof(int64_t));
  if (!parent_coords) {
    tensor_free_cpu(result);
    return NULL;
  }

  for (i = 0; i < grad->numel; i++) {
    grad_val = read_tensor_element(grad, i);

    size_t temp = i;
    view_offset = view->storage_offset;
    for (ptrdiff_t vd = (ptrdiff_t)view->ndim - 1; vd >= 0; vd--) {
      int64_t coord = temp % view->shape[vd];
      temp /= view->shape[vd];
      view_offset += (size_t)(coord * view->strides[vd]);
    }

    offset_to_parent_coords(parent, view_offset, parent_coords);

    result_offset = result->storage_offset;
    for (d = 0; d < parent->ndim; d++) {
      result_offset += (size_t)(parent_coords[d] * result->strides[d]);
    }

    result->storage->data[result_offset] += grad_val;
  }

  free(parent_coords);
  return result;
}
