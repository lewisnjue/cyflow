#include "cyflow/matmul_cpu.h"
#include <stdlib.h>

void tensor_matmul_out_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2) {
  size_t ndim1 = src1->ndim;
  size_t ndim2 = src2->ndim;
  size_t out_ndim = dst->ndim;
  size_t batch_ndim1 = ndim1 - 2;
  size_t batch_ndim2 = ndim2 - 2;
  size_t out_batch_ndim = out_ndim - 2;

  int64_t M = src1->shape[ndim1 - 2];
  int64_t K = src1->shape[ndim1 - 1];
  int64_t N = src2->shape[ndim2 - 1];

  int64_t a_row_stride = src1->strides[ndim1 - 2];
  int64_t a_col_stride = src1->strides[ndim1 - 1];
  int64_t b_row_stride = src2->strides[ndim2 - 2];
  int64_t b_col_stride = src2->strides[ndim2 - 1];
  int64_t d_row_stride = dst->strides[out_ndim - 2];
  int64_t d_col_stride = dst->strides[out_ndim - 1];

  int64_t *a_batch_strides = NULL;
  int64_t *b_batch_strides = NULL;

  if (out_batch_ndim > 0) {
    a_batch_strides = (int64_t *)malloc(out_batch_ndim * sizeof(int64_t));
    b_batch_strides = (int64_t *)malloc(out_batch_ndim * sizeof(int64_t));
    if (!a_batch_strides || !b_batch_strides) {
      if (a_batch_strides) free(a_batch_strides);
      if (b_batch_strides) free(b_batch_strides);
      return;
    }

    for (size_t d = 0; d < out_batch_ndim; d++) {
      /* Same trailing-alignment broadcast rule as compute_broadcast_shape:
       * a dim missing from the smaller-ndim operand, or a size-1 dim,
       * contributes stride 0 (i.e. it gets "reused" for every batch idx). */
      int64_t a_idx = (int64_t)d - (int64_t)(out_batch_ndim - batch_ndim1);
      a_batch_strides[d] =
          (a_idx >= 0 && src1->shape[a_idx] != 1) ? src1->strides[a_idx] : 0;

      int64_t b_idx = (int64_t)d - (int64_t)(out_batch_ndim - batch_ndim2);
      b_batch_strides[d] =
          (b_idx >= 0 && src2->shape[b_idx] != 1) ? src2->strides[b_idx] : 0;
    }
  }

  size_t batch_numel = 1;
  for (size_t d = 0; d < out_batch_ndim; d++) {
    batch_numel *= (size_t)dst->shape[d];
  }

  const float *a_data = src1->storage->data;
  const float *b_data = src2->storage->data;
  float *d_data = dst->storage->data;

  #pragma omp parallel for
  for (size_t bi = 0; bi < batch_numel; bi++) {
    size_t temp = bi;
    int64_t a_off = (int64_t)src1->storage_offset;
    int64_t b_off = (int64_t)src2->storage_offset;
    int64_t d_off = (int64_t)dst->storage_offset;

    for (int d = (int)out_batch_ndim - 1; d >= 0; d--) {
      int64_t dim_size = dst->shape[d];
      int64_t coord = (int64_t)(temp % (size_t)dim_size);
      temp /= (size_t)dim_size;
      a_off += coord * a_batch_strides[d];
      b_off += coord * b_batch_strides[d];
      d_off += coord * dst->strides[d];
    }

    for (int64_t m = 0; m < M; m++) {
      int64_t a_base = a_off + m * a_row_stride;
      for (int64_t n = 0; n < N; n++) {
        int64_t b_base = b_off + n * b_col_stride;
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
          sum += a_data[a_base + k * a_col_stride] * b_data[b_base + k * b_row_stride];
        }
        d_data[d_off + m * d_row_stride + n * d_col_stride] = sum;
      }
    }
  }

  if (a_batch_strides) free(a_batch_strides);
  if (b_batch_strides) free(b_batch_strides);
}
