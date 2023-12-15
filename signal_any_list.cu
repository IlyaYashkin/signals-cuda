#include <iostream>
#include <stdio.h>
#include <cmath>
#include "math_constants.h"

#include <thrust/extrema.h>

using namespace std;

#define N 5
#define PHASE 180
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

  c[blockDim.x * blockIdx.x + threadIdx.x] = akf;
}

int main()
{
  size_t m_trans_size = BASE * sizeof(float);

  float *signal_Re = (float*)malloc(m_trans_size);
  float *signal_Im = (float*)malloc(m_trans_size);

  float *dev_signal_Re;
  float *dev_signal_Im;

  for (int i = 0; i < BASE; i++) {
    float rad = 2 * CUDART_PI * i / BASE;
    signal_Im[i] = sin(rad);
    signal_Re[i] = cos(rad);
  }

  cudaMalloc(&dev_signal_Re, m_trans_size);
  cudaMalloc(&dev_signal_Im, m_trans_size);
  cudaMemcpy(dev_signal_Re, signal_Re, m_trans_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_signal_Im, signal_Im, m_trans_size, cudaMemcpyHostToDevice);


  unsigned long num_combinations = pow(BASE, N) - 2;

  size_t size = N * num_combinations * sizeof(float);

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

  cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);

  int counter = 0;

  for (int i = 0; i < N * num_combinations; i++)
  {
      if (counter == N)
      {
        cout << ' ' << i / N << endl;
        counter = 0;
      }

      printf("%.2f ", host_c[i]);
      counter++;
  }


  free(host_c);
  cudaFree(dev_c);

  return 0;
}
