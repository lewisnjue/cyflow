#include "cyflow/utils.h"


int compute_broadcast_shape(
    const int64_t *shape_a, size_t ndim_a,
    const int64_t *shape_b, size_t ndim_b,
    int64_t *out_shape, size_t *out_ndim) 
{
    // The output dimensions will be the maximum of the two input dimensions
    size_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    // Iterate from right to left (end of shape array to the beginning)
    for (size_t i = 1; i <= max_ndim; i++) {
        // If we run out of dimensions on one side, treat it as 1
        int64_t dim_a = (i <= ndim_a) ? shape_a[ndim_a - i] : 1;
        int64_t dim_b = (i <= ndim_b) ? shape_b[ndim_b - i] : 1;

        if (dim_a == dim_b) {
            out_shape[max_ndim - i] = dim_a;
        } else if (dim_a == 1) {
            out_shape[max_ndim - i] = dim_b;
        } else if (dim_b == 1) {
            out_shape[max_ndim - i] = dim_a;
        } else {
            // Broadcasting failed (e.g., trying to match 3 and 4)
            return -1; 
        }
    }

    return 0; // Success
}