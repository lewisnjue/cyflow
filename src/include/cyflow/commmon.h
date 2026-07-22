#ifndef TENSOR_COMMON_H
#define TENSOR_COMMON_H
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#define MAX_DIM(a, b) ((a) > (b) ? (a) : (b))
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_CUDA = 1
} DeviceType;

typedef struct {
    float *data;          // Pointer to data (can be CPU or GPU pointer)
    size_t size;          // Number of elements
    int ref_count;
    bool owns_data;
    DeviceType device;    // Tracks if data is on CPU or GPU
} Storage;

typedef struct {
    Storage *storage;
    int64_t *shape;       // ALWAYS on the CPU
    int64_t *strides;     // ALWAYS on the CPU
    size_t ndim;
    size_t numel;
    size_t storage_offset;
} TensorImpl;


static inline void compute_contiguous_strides(int64_t *strides, const int64_t *shape, size_t ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}
#endif