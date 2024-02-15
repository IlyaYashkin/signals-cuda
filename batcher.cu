#include <iostream>
#include <stdio.h>
#include <cmath>
#include "math_constants.h"

#include <thrust/extrema.h>

using namespace std;

#define N 11
#define PHASE 30
#define BASE (360 / PHASE)

#define TRANSFORM_MATRIX_ITERATIONS_NUMBER ceil((float) BASE / (float) N)

/* size in bytes */
#define BATCH_SIZE 6442450944
#define BATCH BATCH_SIZE / sizeof(float)

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__global__ void kernel(
  float *c,
  unsigned long long offset
  )
{
  __shared__ float signal_Re[BASE];
  __shared__ float signal_Im[BASE];

  for (int i = 0; i < TRANSFORM_MATRIX_ITERATIONS_NUMBER; i++) {
    int idx = threadIdx.x + i * N;

    if (idx > BASE - 1) { break; }

    float rad = 2 * CUDART_PI * idx / BASE;
    signal_Re[idx] = cosf(rad);
    signal_Im[idx] = sinf(rad);
  }

  __syncthreads();

  __shared__ char signal[N];

  unsigned long long signal_part = blockIdx.x + offset;

  for (int i = 0; i < threadIdx.x; i++) {
    if (signal_part == 0) { break; }
    signal_part /= BASE;
  }

  signal[threadIdx.x] = signal_part % BASE;

  __syncthreads();

  if (threadIdx.x == 0) {
    return;
  }

  __shared__ float max;
  max = 0;

  float sum_Re = 0.0;
  float sum_Im = 0.0;
  for (int i = 0; i + threadIdx.x < N; i++) {
    int idx = (BASE + signal[i] - signal[threadIdx.x + i]) % BASE;

    sum_Re += signal_Re[idx];
    sum_Im += signal_Im[idx];
  }

  float akf = sqrtf(sum_Re * sum_Re + sum_Im * sum_Im);

  atomicMaxFloat(&max, akf);

  if (threadIdx.x != 1) {
    return;
  }

  c[blockIdx.x] = max;
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
  
  int threadsPerBlock = N;
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

unsigned long long get_num_combinations()
{
  unsigned long long num_combinations = BASE;

  for (int i = 0; i < N; i++) {
    num_combinations = num_combinations * BASE;
  }

  return num_combinations;
}

int main()
{
  unsigned long long num_combinations = get_num_combinations();
  size_t size = num_combinations * sizeof(float);

  if (size <= 0) {
    cout << "result array size error" << endl;
    return 1;
  }

  unsigned long long num_batches = ceil((double) size / BATCH_SIZE);

  printf("BATCH COUNT: %lld\n", num_batches);

  unsigned long long start_from = 0;
  // unsigned long long start_from = 1031655766;

  // unsigned long long result;

  for (unsigned long long i = start_from; i < num_batches; i++) {
    unsigned long long offset = i * BATCH;

    long long comp = (num_combinations - (offset + BATCH));

    unsigned long long batch = comp < 0 ? num_combinations - offset : BATCH;

    printf("\n --- BATCH %lld --- \n\n", i);

    unsigned long long batch_result = start_kernel(offset, batch);
  }
}