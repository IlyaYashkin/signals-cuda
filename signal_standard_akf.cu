#include <iostream>
#include <stdio.h>
#include <cmath>
#include "math_constants.h"

#include <thrust/extrema.h>

using namespace std;

typedef float2 Complex;

#define N 60
#define PHASE 180
#define BASE (360 / PHASE)

#define TRANSITION_MATRIX_ITERATIONS_NUMBER ceil((float) BASE / (float) N)

/* size in bytes */
#define BATCH_SIZE 6442450944
#define BATCH BATCH_SIZE / sizeof(float)

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

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
  unsigned long long offset
)
{
  unsigned long long signal_part = blockIdx.x + offset;

  for (int i = 0; i < threadIdx.x; i++) {
    if (signal_part == 0) { break; }
    signal_part /= BASE;
  }

  unsigned int t_idx = signal_part % BASE;

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

__global__ void kernel(
  float *c,
  unsigned long long offset
  )
{
  if (threadIdx.x >= N) { return; }

  __shared__ Complex transition_matrix[BASE];

  getTransitionMatrix(transition_matrix);

  __syncthreads();

  __shared__ Complex signal[N];

  getSignal(signal, transition_matrix, offset);

  __syncthreads();

  if (threadIdx.x == 0) {
    return;
  }

  float akf = findAkf(signal);

  __shared__ float max;
  max = 0;

  atomicMaxFloat(&max, akf);

  if (threadIdx.x != 1) {
    return;
  }

  c[blockIdx.x] = max;
}

unsigned long upperPowerOfTwo(unsigned long v)
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

unsigned long long start_kernel(
  unsigned long long offset,
  unsigned long long batch
  )
{
  unsigned long long batch_size = batch * sizeof(float);

  float *host_c = (float*)malloc(batch_size);
  float host_akf;

  float *dev_c;

  cudaMalloc(&dev_c, batch_size);
  
  int threadsPerBlock = upperPowerOfTwo(N);
  unsigned long long blocksInGrid = batch;

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
  unsigned long long result = thrust::min_element(thrust::device, dev_c, dev_c + batch) - dev_c;
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(&host_akf, dev_c + result, sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&t, start, stop);
  printf("thrust::min_element time: %f\n", t);

  printf("best signal: %zd\n", result + offset);
  printf("akf: %f\n", host_akf);

  free(host_c);
  cudaFree(dev_c);

  return 0;
}

unsigned long long getNumCombinations()
{
  unsigned long long num_combinations = BASE;

  for (int i = 0; i < N; i++) {
    num_combinations = num_combinations * BASE;
  }

  return num_combinations;
}

int main()
{
  unsigned long long num_combinations = getNumCombinations();
  size_t size = num_combinations * sizeof(float);

  if (size <= 0) {
    cout << "result array size error" << endl;
    return 1;
  }

  unsigned long long num_batches = ceil((double) size / BATCH_SIZE);

  printf("BATCH COUNT: %lld\n", num_batches);

  unsigned long long start_from = 0;

  for (unsigned long long i = start_from; i < num_batches; i++) {
    unsigned long long offset = i * BATCH;

    long long comp = (num_combinations - (offset + BATCH));

    unsigned long long batch = comp < 0 ? num_combinations - offset : BATCH;

    printf("\n --- BATCH %lld --- \n\n", i);

    unsigned long long batch_result = start_kernel(offset, batch);
  }
}