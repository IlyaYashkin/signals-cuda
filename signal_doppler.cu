#include <iostream>
#include <inttypes.h>
#include <stdio.h>

#include "math_constants.h"
#include <thrust/extrema.h>

#include "utils/calc.h"
#include "utils/error.h"

using namespace std;

typedef float2 Complex;

#define N 30
#define PHASE 180
#define BASE (360 / PHASE)

#define TRANSITION_MATRIX_ITERATIONS_NUMBER (int) ceilf((float) BASE / (float) N)

/* size in bytes */
#define CHUNK_SIZE_IN_BYTES 6442450944
#define CHUNK_SIZE CHUNK_SIZE_IN_BYTES / sizeof(float)

uint64_t start_kernel(
  uint64_t offset,
  uint64_t chunk_size
  )
{
  uint64_t chunk_size_m = chunk_size * sizeof(float);

  float *host_c = (float*)malloc(chunk_size_m);
  float host_akf;

  float *dev_c;

  cudaMalloc(&dev_c, chunk_size_m);
  
  int threadsPerBlock = upperPowerOfTwo(N);
  uint64_t blocksInGrid = chunk_size;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel<<< blocksInGrid, threadsPerBlock >>>(dev_c, offset);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float t;

  cudaEventElapsedTime(&t, start, stop);
  printf("gpu time: %f\n", t);

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  uint64_t result = thrust::max_element(thrust::device, dev_c, dev_c + chunk_size) - dev_c;
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(&host_akf, dev_c + result, sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&t, start, stop);
  printf("thrust::min_element time: %f\n", t);

  cout << "best signal: " << result + offset << endl;
  printf("akf: %f\n", host_akf);

  free(host_c);
  cudaFree(dev_c);

  return 0;
}

int main()
{
  uint64_t combinations_count = getCombinationsCount();
  uint64_t chunk_size = CHUNK_SIZE_IN_BYTES / sizeof(float);

  if (combinations_count <= 0) {
    printf("result array size error\n");
    return 1;
  }

  uint64_t chunks_count = 
    combinations_count < chunk_size ? 1 : combinations_count / chunk_size;

  cout << "CHUNKS COUNT: " << chunks_count << endl;

  uint64_t start_from = 0;

  for (uint64_t i = start_from; i < chunks_count; i++) {
    uint64_t offset = i * chunk_size;

    int64_t comp = (combinations_count - (offset + chunk_size));

    uint64_t residual_chunk_size = comp < 0 ? combinations_count - offset : chunk_size;

    printf("\n --- CHUNK %" PRIu64 " --- \n\n", i);

    start_kernel(offset, residual_chunk_size);

    break;
  }
}
