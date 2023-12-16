#include <iostream>
#include <stdio.h>
#include <cmath>
#include "math_constants.h"

#include <thrust/extrema.h>

using namespace std;

#define N 6
#define PHASE 90
#define BASE (360 / PHASE)

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__global__ void kernel(float *c, float *signal_Re, float *signal_Im)
{
  __shared__ char signal[N];

  int signal_part = blockIdx.x + 1;

  for (int i = 0; i < threadIdx.x; i++) {
    if (signal_part == 0) { break; }
    signal_part /= BASE;
  }

  signal[threadIdx.x] = signal_part % BASE;

  if (threadIdx.x == 0) {
    return;
  }

  __shared__ float max;
  max = 0.0;

  float sum_Re = 0;
  float sum_Im = 0;
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

int main()
{
  size_t m_trans_size = BASE * sizeof(float);

  float *host_signal_Re = (float*)malloc(m_trans_size);
  float *host_signal_Im = (float*)malloc(m_trans_size);

  float *dev_signal_Re;
  float *dev_signal_Im;

  for (int i = 0; i < BASE; i++) {
    float rad = 2 * CUDART_PI * i / BASE;
    host_signal_Im[i] = sin(rad);
    host_signal_Re[i] = cos(rad);
  }

  cudaMalloc(&dev_signal_Re, m_trans_size);
  cudaMalloc(&dev_signal_Im, m_trans_size);
  cudaMemcpy(dev_signal_Re, host_signal_Re, m_trans_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_signal_Im, host_signal_Im, m_trans_size, cudaMemcpyHostToDevice);


  unsigned long num_combinations = pow(BASE, N) - 2;

  size_t size = num_combinations * sizeof(float);

  float *host_c = (float*)malloc(size);

  float *dev_c;

  cudaMalloc(&dev_c, size);
  
  int threadsPerBlock = N;
  unsigned long blocksInGrid = num_combinations;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel<<< blocksInGrid, threadsPerBlock >>>(dev_c, dev_signal_Re, dev_signal_Im);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float t;

  cudaEventElapsedTime(&t, start, stop);
  printf("gpu time: %f\n", t);

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  int result = thrust::min_element(thrust::device, dev_c, dev_c + num_combinations) - dev_c;
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&t, start, stop);
  printf("thrust::min_element time: %f\n", t);

  printf("best signal: %d\n", result + 1);

  free(host_c);
  free(host_signal_Re);
  free(host_signal_Im);
  cudaFree(dev_c);
  cudaFree(dev_signal_Re);
  cudaFree(dev_signal_Im);

  return 0;
}
