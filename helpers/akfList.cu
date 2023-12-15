#include <iostream>
#include <cmath>
#include "./akfuncdemo/akfuncdemo.h"

using namespace std;

#define N 5

__global__ void kernel(int *c)
{
  __shared__ char signal[N];
  signal[threadIdx.x] = blockIdx.x + 1 & 1 << threadIdx.x ? 1 : -1;

  if (threadIdx.x == 0) {
    return;
  }

  int akf = 0;
  for (int i = 0; i + threadIdx.x < N; i++) {
    akf += signal[threadIdx.x + i] * signal[i];
  }

  akf = abs(akf);

  c[blockDim.x * blockIdx.x + threadIdx.x] = akf;
}

int main()
{
  int num_combinations = pow(2, N) - 2;

  size_t size = N * num_combinations * sizeof(int);

  int *C = (int*)malloc(size);

  int *dev_C;

  cudaMalloc(&dev_C, size);
  
  int threadsPerBlock = N;
  int blocksInGrid = num_combinations;

  kernel<<< blocksInGrid, threadsPerBlock >>>(dev_C);

  cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

  int counter = 0;

  for (int i = 0; i < N * num_combinations; i++)
  {
      if (counter == N)
      {
        cout << ' ' << i / N << endl;
        counter = 0;
      }
      cout << C[i];
      counter++;
  }

  free(C);

  cudaFree(dev_C);

  return 0;
}
