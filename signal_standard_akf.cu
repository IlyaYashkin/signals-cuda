#include <iostream>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>

#include <omp.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "utils/calc.h"
#include "utils/error.h"
#include "signal/signal.h"

using namespace std;

#define N 31
#define PHASE 180
#define BASE (360 / PHASE)

#define CHUNK_SIZE_IN_BYTES 5368709120

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
  
  int threadsPerBlock = upperPowerOfTwo(signal_size);
  uint64_t blocksInGrid = chunk_size;

  size_t shared_memory_size = 
    base * sizeof(float2) +        // transition matrix
    signal_size * sizeof(float2);  // signal array

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel<<< blocksInGrid, threadsPerBlock, shared_memory_size >>>(dev_c, offset, signal_size, base);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float t;

  cudaEventElapsedTime(&t, start, stop);
  printf("gpu time: %f\n", t);

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  uint64_t result = thrust::min_element(thrust::device, dev_c, dev_c + chunk_size) - dev_c;
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

uint64_t getResidualChunkSize(uint64_t offset, uint64_t chunk_size, uint64_t combinations_count)
{
  int64_t comp = (combinations_count - (offset + chunk_size));
  return comp < 0 ? combinations_count - offset : chunk_size;
}

uint64_t getOffset(uint64_t i, uint64_t chunk_size)
{
  return i * chunk_size;
}

float* start_kernel_async(
  cudaStream_t stream,
  uint64_t offset,
  uint64_t chunk_size,
  uint32_t signal_size,
  uint32_t base
)
{
  uint64_t chunk_size_m = chunk_size * sizeof(float);

  float *dev_c;

  cudaMallocAsync(&dev_c, chunk_size_m, stream);
  
  int threadsPerBlock = upperPowerOfTwo(signal_size);
  uint64_t blocksInGrid = chunk_size;
  size_t shared_memory_size = 
    base * sizeof(float2) +        // transition matrix
    signal_size * sizeof(float2);  // signal array

  kernel<<< blocksInGrid, threadsPerBlock, shared_memory_size, stream >>>(dev_c, offset, signal_size, base);

  return dev_c;
}

int main()
{
  uint64_t combinations_count = getCombinationsCount(N, BASE);
  uint64_t chunk_size = CHUNK_SIZE_IN_BYTES / sizeof(float);
  uint32_t signal_size = N;
  uint32_t base = BASE;
  uint64_t start_from = 0;

  if (checkOverflow(combinations_count)) { return 1; }

  uint64_t chunk_count = 
    combinations_count < chunk_size ? 1 : combinations_count / chunk_size;

  cout << "CHUNKS COUNT: " << chunk_count << endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  omp_set_num_threads(device_count);

  #pragma omp parallel for schedule(dynamic)
  for (uint64_t i = start_from; i < chunk_count; i++) {
    int device = omp_get_thread_num();

    cudaSetDevice(device);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint64_t offset = getOffset(i, chunk_size);
    uint64_t residual_chunk_size = getResidualChunkSize(offset, chunk_size, combinations_count);

    printf(" --- Device %d: CHUNK %" PRIu64 " --- \n", device, i);

    float *dev_c = start_kernel_async(
      stream,
      offset,
      residual_chunk_size,
      signal_size,
      base
    );

    cudaStreamSynchronize(stream);

    uint64_t result = thrust::min_element(thrust::device.on(stream), dev_c, dev_c + residual_chunk_size) - dev_c;

    float test;

    cudaMemcpyAsync(&test, dev_c + result, sizeof(float), cudaMemcpyDeviceToHost, stream);

    cout << "device: " << device << endl;
    cout << "akf: " << test << endl;
    cout << "signal: " << result + offset << endl;

    cudaFree(dev_c);
  }
}