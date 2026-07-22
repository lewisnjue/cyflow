#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"

Storage *storage_create_cuda(size_t size) {
    (void)size;
    return NULL;
}

void storage_free_cuda(Storage *storage) {
    (void)storage;
}

TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim) {
    (void)shape;
    (void)ndim;
    return NULL;
}

void tensor_free_cuda(TensorImpl *tensor) {
    (void)tensor;
}

void tensor_fill_uniform_cuda(TensorImpl *tensor) {
    (void)tensor;
}

void cyflow_manual_seed_cuda(unsigned long long seed) {
    (void)seed;
}
