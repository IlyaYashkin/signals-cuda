#include <iostream>
#include <inttypes.h>
#include <stdio.h>

#include "math_constants.h"
#include <thrust/extrema.h>

#include "utils/calc.h"
#include "utils/error.h"
#include "signal/signal.h"

using namespace std;

#define N 30
#define PHASE 180
#define BASE (360 / PHASE)

/* size in bytes */
#define CHUNK_SIZE_IN_BYTES 6442450944

uint64_t start_kernel(
  uint64_t offset,
  uint64_t chunk_size,
  uint32_t signal_size,
  uint32_t base
)
{
  uint64_t chunk_size_m = chunk_size * sizeof(float);

  float *host_c = (float*)malloc(chunk_size_m);
  float host_akf;

  float *dev_c;

  cudaMalloc(&dev_c, chunk_size_m);

  size_t shared_memory_size = 
    base * sizeof(float2) +         // transition matrix
    signal_size * sizeof(float2) +  // signal array
    signal_size * sizeof(float2);   // signal array (copy)
  
  int threadsPerBlock = upperPowerOfTwo(signal_size);
  uint64_t blocksInGrid = chunk_size;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel_doppler<<< blocksInGrid, threadsPerBlock, shared_memory_size >>>(dev_c, offset, signal_size, base);
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
  uint64_t combinations_count = getCombinationsCount(N, BASE);
  uint64_t chunk_size = CHUNK_SIZE_IN_BYTES / sizeof(float);
  uint32_t signal_size = N;
  uint32_t base = BASE;
  uint64_t start_from = 0;

  if (checkOverflow(combinations_count)) { return 1; }

  uint64_t chunks_count = 
    combinations_count < chunk_size ? 1 : combinations_count / chunk_size;

  cout << "CHUNKS COUNT: " << chunks_count << endl;

  for (uint64_t i = start_from; i < chunks_count; i++) {
    uint64_t offset = i * chunk_size;

    int64_t comp = (combinations_count - (offset + chunk_size));

    uint64_t residual_chunk_size = comp < 0 ? combinations_count - offset : chunk_size;

    printf("\n --- CHUNK %" PRIu64 " --- \n\n", i);

    start_kernel(offset, residual_chunk_size, signal_size, base);
  }
}
