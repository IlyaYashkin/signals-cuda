#include <iostream>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
#include "math_constants.h"

#include <thrust/extrema.h>

using namespace std;

typedef float2 Complex;

#define N 61
#define PHASE 180
#define BASE (360 / PHASE)

#define TRANSITION_MATRIX_ITERATIONS_NUMBER (int) ceilf((float) BASE / (float) N)

/* size in bytes */
#define CHUNK_SIZE_IN_BYTES 6442450944
#define CHUNK_SIZE CHUNK_SIZE_IN_BYTES / sizeof(float)

__device__ void getTransitionMatrix(Complex* transition_matrix)
{
  for (int i = 0; i < TRANSITION_MATRIX_ITERATIONS_NUMBER; i++) {
    int idx = threadIdx.x + i * N;

    if (idx > BASE - 1) { break; }

    float rad = 2 * CUDART_PI * idx / BASE;

    transition_matrix[idx].x = cosf(rad);
    transition_matrix[idx].y = sinf(rad);
  }
}

__device__ void getSignal(
  Complex* signal,
  Complex* transition_matrix,
  uint64_t offset
)
{
  if (threadIdx.x >= N) { return; }

  uint64_t signal_part = blockIdx.x + offset;

  for (int i = 0; i < threadIdx.x && threadIdx.x < N; i++) {
    if (signal_part == 0) { break; }
    signal_part /= BASE;
  }

  uint32_t t_idx = signal_part % BASE;

  signal[threadIdx.x].x = transition_matrix[t_idx].x;
  signal[threadIdx.x].y = transition_matrix[t_idx].y;
}

__device__ float findAkf(Complex* signal)
{
  Complex sum = {0.0, 0.0};

  for (int i = 0; i + threadIdx.x < N; i++) {
    sum.x += signal[threadIdx.x + i].x * signal[i].x - signal[threadIdx.x + i].y * -signal[i].y;
    sum.y += signal[threadIdx.x + i].x * -signal[i].y + signal[threadIdx.x + i].y * signal[i].x;
  }

  return sqrtf(sum.x * sum.x + sum.y * sum.y);
}

__device__ void reduceMax(Complex* signal)
{
  uint32_t i = blockDim.x / 2;

  while (i != 0) {
    if (threadIdx.x <= i && threadIdx.x + i < N) {
      signal[threadIdx.x].x = fmaxf(signal[threadIdx.x].x, signal[threadIdx.x + i].x);
    }

    __syncthreads();

    i /= 2;
  }
}

__global__ void kernel(
  float *c,
  uint64_t offset
  )
{
  __shared__ Complex transition_matrix[BASE];

  getTransitionMatrix(transition_matrix);

  __syncthreads();

  __shared__ Complex signal[N];

  getSignal(signal, transition_matrix, offset);

  __syncthreads();

  float akf = 0;

  if (threadIdx.x != 0) {
    akf = findAkf(signal);
  }

  __syncthreads();

  if (threadIdx.x < N) {
    signal[threadIdx.x].x = akf;
  }

  __syncthreads();

  reduceMax(signal);

  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  c[blockIdx.x] = signal[0].x;
}

uint32_t upperPowerOfTwo(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

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

uint64_t getCombinationsCount()
{
  uint64_t combinations_count = BASE;

  for (int i = 0; i < N; i++) {
    combinations_count = combinations_count * BASE;
  }

  return combinations_count;
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

  uint64_t start_from = 1031655765;

  for (uint64_t i = start_from; i < chunks_count; i++) {
    uint64_t offset = i * chunk_size;

    int64_t comp = (combinations_count - (offset + chunk_size));

    uint64_t residual_chunk_size = comp < 0 ? combinations_count - offset : chunk_size;

    printf("\n --- CHUNK %" PRIu64 " --- \n\n", i);

    start_kernel(offset, residual_chunk_size);
  }
}