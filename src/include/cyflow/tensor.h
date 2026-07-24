#ifndef TENSOR_H
#define TENSOR_H

#include "cyflow/common.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DeviceType { 
    DEVICE_CPU = 0, 
    DEVICE_CUDA = 1 
} DeviceType;

typedef struct Storage {
  float *data; // Pointer to data (can be CPU or GPU pointer)
  size_t size; // Number of elements
  int ref_count;
  bool owns_data;
  DeviceType device; // Tracks if data is on CPU or GPU
} Storage;

typedef struct TensorImpl {
  Storage *storage;
  int64_t *shape;   // ALWAYS on the CPU
  int64_t *strides; // ALWAYS on the CPU
  size_t ndim;
  size_t numel;
  size_t storage_offset;
} TensorImpl;

typedef struct TensorMeta {
  int64_t shape[MAX_DIMS];
  int64_t strides[MAX_DIMS];
  size_t ndim;
  size_t storage_offset;
} TensorMeta;

TensorMeta create_tensor_meta(const TensorImpl *t);

bool tensor_is_contiguous(const TensorImpl *tensor);
void compute_contiguous_strides(int64_t *strides, const int64_t *shape,
                                size_t ndim);

TensorImpl *tensor_view(TensorImpl *src, const int64_t *shape, size_t ndim);

TensorImpl *tensor_index(TensorImpl *src, int64_t index);
Storage *storage_create_cpu(size_t size);
void storage_free_cpu(Storage *storage);

TensorImpl *tensor_create_cpu(const int64_t *shape, size_t ndim);
void tensor_free_cpu(TensorImpl *tensor);

void cyflow_manual_seed_cpu(unsigned long long seed);
void tensor_fill_uniform_cpu(TensorImpl *tensor);
void tensor_set_data_cpu(TensorImpl *tensor, const float *data);

#ifdef __cplusplus
}
#endif

#endif