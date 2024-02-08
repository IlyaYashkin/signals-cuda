#include <iostream>
#include <stdio.h>
#include <cmath>

#include <thrust/extrema.h>

using namespace std;

#define N 10

__global__ void kernel(unsigned short *c)
{
  __shared__ char signal[N];
  signal[threadIdx.x] = blockIdx.x + 1 & 1 << threadIdx.x ? 1 : -1;

  if (threadIdx.x == 0) {
    return;
  }

  __shared__ unsigned int max;
  max = 0;

  int akf = 0;
  for (int i = 0; i + threadIdx.x < N; i++) {
    akf += signal[threadIdx.x + i] * signal[i];
  }

  akf = abs(akf);

  atomicMax(&max, akf);

  if (threadIdx.x != 1) {
    return;
  }

  c[blockIdx.x] = max;
}

int main()
{
  unsigned long num_combinations = pow(2, N) - 2;

  size_t size = num_combinations * sizeof(short);

  unsigned short *host_c = (unsigned short*)malloc(size);

  unsigned short *dev_c;

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
