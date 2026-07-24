#ifndef CYFLOW_UTILS_H
#define CYFLOW_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


int compute_broadcast_shape(const int64_t *shape_a, size_t ndim_a,
                            const int64_t *shape_b, size_t ndim_b,
                            int64_t *out_shape, size_t *out_ndim);

#ifdef __cplusplus
}
#endif

#endif // CYFLOW_UTILS_H