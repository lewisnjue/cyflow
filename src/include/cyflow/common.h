#ifndef TENSOR_COMMON_H
#define TENSOR_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIM(a, b) ((a) > (b) ? (a) : (b))

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuRAND Error: %d at %s:%d\n", (int)err_, __FILE__,      \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#ifndef MAX_DIMS
#define MAX_DIMS 8
#endif

#define CUDA_THREADS_PER_BLOCK 256

#endif
