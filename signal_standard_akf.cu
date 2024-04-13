#include <iostream>
#include <sstream>
#include <fstream>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>

#include <cstdlib>

#include <omp.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "utils/calc.h"
#include "utils/error.h"
#include "signal/signal.h"

using namespace std;

#define N 40
#define PHASE 180
#define BASE (360 / PHASE)

#define CHUNK_SIZE_IN_BYTES 5368709120
// #define CHUNK_SIZE_IN_BYTES 4096

struct radar_signal
{
  float akf;
  uint64_t signal;
};

radar_signal best_signal = { akf: FLT_MAX };

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
  uint64_t chunk_start = 0;
  uint8_t chunk_step = 10;

  if (checkOverflow(combinations_count)) { return 1; }

  uint64_t chunk_count = 
    combinations_count < chunk_size ? 1 : combinations_count / chunk_size;

  cout << "CHUNKS COUNT: " << chunk_count << endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  omp_set_num_threads(device_count);

  while (chunk_start < chunk_count) {
    if (chunk_start + chunk_step > chunk_count) {
      chunk_step = chunk_count - chunk_start;
    }

    radar_signal signal_arr[chunk_step];

    #pragma omp parallel for schedule(dynamic)
    for (uint64_t i = chunk_start; i < chunk_start + chunk_step; i++) {
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

      float host_c;

      cudaMemcpyAsync(&host_c, dev_c + result, sizeof(float), cudaMemcpyDeviceToHost, stream);

      signal_arr[i - chunk_start] = { host_c, result + offset };

      cudaFree(dev_c);
    }

    printf("chunks processed %d\n", chunk_step);

    radar_signal chunk_best_signal = *min_element(
      signal_arr,
      signal_arr + chunk_step,
      [](radar_signal& a, radar_signal& b) {
        return a.akf < b.akf;
      });

    if (chunk_best_signal.akf < best_signal.akf) {
      best_signal = chunk_best_signal;

      ostringstream file_name;
      file_name << "./signals/base" << base << "_signal" << signal_size << endl;

      ofstream signal_file(file_name.str());

      signal_file << "signal: "   << best_signal.signal   << endl;
      signal_file << "akf: "      << best_signal.akf      << endl;

      signal_file.close();
    }

    chunk_start += chunk_step;

  }

  cout << "akf: "     << best_signal.akf      << endl;
  cout << "signal: "  << best_signal.signal   << endl;
}