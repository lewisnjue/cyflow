#include "cyflow/common.h"
#include "cyflow/tensor_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>


#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define CURAND_CHECK(err) \
    do { \
        curandStatus_t err_ = (err); \
        if (err_ != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND Error: %d at %s:%d\n", (int)err_, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

static curandGenerator_t cuda_gen;
static bool curand_initialized = false;


void cyflow_manual_seed_cuda(unsigned long long seed) {
    if (!curand_initialized) {
        CURAND_CHECK(curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curand_initialized = true;
    }
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cuda_gen, seed));
}

void tensor_fill_uniform_cuda(TensorImpl *tensor) {
    if (!tensor || !tensor->storage || !tensor->storage->data) return;
    if (!curand_initialized) {
        cyflow_manual_seed_cuda(42ULL);
    }
    CURAND_CHECK(curandGenerateUniform(cuda_gen, tensor->storage->data, tensor->numel));
}

Storage *storage_create_cuda(size_t size) {
    Storage *storage = (Storage *)malloc(sizeof(Storage));
    if (!storage) return NULL;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&storage->data, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(storage->data, 0, size * sizeof(float)));

    storage->size = size;
    storage->ref_count = 1;
    storage->owns_data = true;
    storage->device = DEVICE_CUDA;
    return storage;
}

void storage_free_cuda(Storage *storage) {
    if (!storage) return;
    storage->ref_count--;
    if (storage->ref_count == 0) {
        if (storage->owns_data && storage->data) {
            cudaFree(storage->data); // CUDA Free
        }
        free(storage);
    }
}

TensorImpl *tensor_create_cuda(const int64_t *shape, size_t ndim) {
    TensorImpl *tensor = (TensorImpl *)malloc(sizeof(TensorImpl));
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
    tensor->strides = (int64_t *)malloc(ndim * sizeof(int64_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
    compute_contiguous_strides(tensor->strides, tensor->shape, ndim);
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_size *= shape[i];
    }
    tensor->numel = total_size;
    tensor->storage_offset = 0;

    tensor->storage = storage_create_cuda(total_size);
    if (!tensor->storage) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    return tensor;
}