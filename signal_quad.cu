#include <iostream>
#include <stdio.h>
#include <cmath>
#include "./akfuncdemo/akfuncdemo.h"

#include <thrust/extrema.h>

using namespace std;

#define N 6

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__global__ void kernel(float *c)
{
  char signal_Re[4]{1, 0, -1, 0};
  char signal_Im[4]{0, 1, 0, -1};

  __shared__ char signal[N];
  signal[threadIdx.x] = (blockIdx.x + 1 >> (threadIdx.x * 2)) % 4;

  if (threadIdx.x == 0) {
    return;
  }

  __shared__ float max;
  max = 0.0;

  int sum_Re = 0;
  int sum_Im = 0;
  for (int i = 0; i + threadIdx.x < N; i++) {
    int idx = (4 + signal[i] - signal[threadIdx.x + i]) % 4;

    sum_Re += signal_Re[idx];
    sum_Im += signal_Im[idx];
  }

  float akf = sqrtf(sum_Re * sum_Re + sum_Im * sum_Im);

  akf = abs(akf);

  atomicMaxFloat(&max, akf);

  if (threadIdx.x != 1) {
    return;
  }

  c[blockIdx.x] = max;
}

int main()
{
  unsigned long num_combinations = pow(4, N) - 2;

  size_t size = num_combinations * sizeof(float);

  float *host_c = (float*)malloc(size);

  float *dev_c;

  cudaMalloc(&dev_c, size);
  
  int threadsPerBlock = N;
  unsigned long blocksInGrid = num_combinations;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel<<< blocksInGrid, threadsPerBlock >>>(dev_c);
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
  cudaFree(dev_c);

  return 0;
}
