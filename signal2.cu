#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>

using namespace std;

#define N 32

__global__ void generateSignals(unsigned int *c, unsigned int n, unsigned long num_digits)
{
  unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned long num = idx / n;

  unsigned long order = idx - n * num;

  if (idx > num_digits - 1)
  {
    return;
  }

  c[idx] = num & 1 << order ? 1 : 0;
}

__global__ void findAkf(unsigned int *c, unsigned int n, unsigned long num_digits)
{
  unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

  int akf = 0;
  for (int i = 0; i + threadIdx.x < N; i++) {
    akf += c[threadIdx.x + i] * c[i];
  }

  akf = abs(akf);

  // c[blockIdx.x] = max;
}

int main()
{
  unsigned int n = N;

  unsigned long num_combinations = pow(2, n);

  size_t size = num_combinations * n * sizeof(unsigned int);

  unsigned int *c = (unsigned int*)malloc(size);

  unsigned int *dev_c;

  cudaMalloc(&dev_c, size);

  unsigned int threadsPerBlock = 512;
  unsigned int blocksInGrid = ceil(num_combinations * n / (double) threadsPerBlock);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  generateSignals<<< blocksInGrid, threadsPerBlock >>>(dev_c, n, num_combinations * n);
  findAkf<<< blocksInGrid, threadsPerBlock >>>(dev_c, n, num_combinations * n);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float t;

  cudaEventElapsedTime(&t, start, stop);
  printf("gpu time: %f\n", t);

  // cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  // ofstream file;

  // file.open("numbers.txt");

  // int counter = 0;

  // for (long i = 0; i < num_combinations * n; i++)
  // {
  //     if (counter == n)
  //     {
  //       file << ' ' << i / n - 1 << endl;
  //       counter = 0;
  //     }
  //     file << c[i];
  //     counter++;
  // }

  // file.close();

  free(c);
  cudaFree(dev_c);

  return 0;
}
