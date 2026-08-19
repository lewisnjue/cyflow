#ifndef MATMUL_CPU_H
#define MATMUL_CPU_H

#include "cyflow/common.h"
#include "cyflow/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Batched matrix multiplication: dst = src1 @ src2
 *
 * src1 shape: (..., M, K)
 * src2 shape: (..., K, N)
 * dst  shape: (broadcast(...), M, N)
 *
 * `dst` must already be allocated by the caller with the correctly
 * broadcasted batch shape (see compute_broadcast_shape). Both src1 and
 * src2 must have ndim >= 2. Leading "batch" dimensions (everything except
 * the trailing two) are broadcast together the same way ordinary
 * elementwise ops broadcast. All three tensors may be arbitrarily strided
 * (e.g. transposed views), which is what makes this safe to feed
 * transposed operands directly for the backward pass.
 */
void tensor_matmul_out_cpu(TensorImpl *dst, const TensorImpl *src1, const TensorImpl *src2);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_CPU_H
