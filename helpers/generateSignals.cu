#include <iostream>
#include <cmath>

using namespace std;

#define N 5

__global__ void kernel(int *c)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  c[idx] = blockIdx.x + 1 & 1 << threadIdx.x ? 1 : 0;
}

int main()
{
  int num_combinations = pow(2, N) - 2;

  size_t size = N * num_combinations * sizeof(int);

  int *host_c = (int*)malloc(size);

  int *dev_c;

  cudaMalloc(&dev_c, size);
  
  int threadsPerBlock = N;
  int blocksInGrid = num_combinations;

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

  cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);

  int counter = 0;

  for (int i = 0; i < N * num_combinations; i++)
  {
      if (counter == N)
      {
        cout << ' ' << i / N << endl;
        counter = 0;
      }
      cout << host_c[i];
      counter++;
  }

  free(host_c);

  cudaFree(dev_c);

  return 0;
}
