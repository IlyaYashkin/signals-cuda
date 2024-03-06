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

#define SIGNAL_TEST 23

#define TRANSITION_MATRIX_ITERATIONS_NUMBER ceil((float) BASE / (float) N)

/* size in bytes */
#define BATCH_SIZE 6442450944
#define BATCH BATCH_SIZE / sizeof(float)

__device__ Complex multiplyComplex(Complex first, Complex second)
{
  return Complex {
    first.x * second.x - first.y * second.y,
    first.x * second.y + first.y * second.x
  };
}

__device__ Complex sumComplex(Complex first, Complex second)
{
  return Complex {
    first.x + second.x,
    first.y + second.y
  };
}

__device__ Complex subComplex(Complex first, Complex second)
{
  return Complex {
    first.x - second.x,
    first.y - second.y
  };
}

__device__ float absComplex(Complex num)
{
  return sqrtf(num.x * num.x + num.y * num.y);
}

__device__ unsigned int reverseBits(unsigned int num, unsigned int numberOfBits)
{
    unsigned int reverse_num = 0;
    int i;
    for (i = 0; i < numberOfBits; i++) {
        if ((num & (1 << i)))
            reverse_num |= 1 << ((numberOfBits - 1) - i);
    }

    return reverse_num;
}

__device__ Complex getConj(Complex num) {
  num.y = -num.y;

  return num;
}

__device__ void multiplyConj(Complex* signal)
{
  signal[threadIdx.x] = multiplyComplex(signal[threadIdx.x], getConj(signal[threadIdx.x]));
  signal[threadIdx.x + blockDim.x] = multiplyComplex(signal[threadIdx.x + blockDim.x], getConj(signal[threadIdx.x + blockDim.x]));
}




__device__ void getTransitionMatrix(Complex* transition_matrix)
{
  for (int i = 0; i < TRANSITION_MATRIX_ITERATIONS_NUMBER; i++) {
    int idx = threadIdx.x + i * N;

    if (idx > BASE - 1) { break; }

    float rad = 2 * CUDART_PI_F * idx / BASE;
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

  if (threadIdx.x < N) {
    for (int i = 0; i < threadIdx.x; i++) {
      if (signal_part == 0) { break; }
      signal_part /= BASE;
    }

    unsigned int t_idx = signal_part % BASE;

    signal[threadIdx.x].x = transition_matrix[t_idx].x;
    signal[threadIdx.x].y = transition_matrix[t_idx].y;
  } else {
    signal[threadIdx.x].x = 0;
    signal[threadIdx.x].y = 0;
  }


  signal[threadIdx.x + blockDim.x].x = 0;
  signal[threadIdx.x + blockDim.x].y = 0;
}




__device__ void myFFT(Complex* signal, unsigned int bitCount, int direction = -1)
{
  unsigned int j = reverseBits(threadIdx.x, bitCount);

  if (threadIdx.x < j) {
      Complex temp = signal[threadIdx.x];
      signal[threadIdx.x] = signal[j];
      signal[j] = temp;
  }

  j = reverseBits(threadIdx.x + blockDim.x, bitCount);

  if (threadIdx.x + blockDim.x < j) {
      Complex temp = signal[threadIdx.x + blockDim.x];
      signal[threadIdx.x + blockDim.x] = signal[j];
      signal[j] = temp;
  }

  __syncthreads();

  unsigned int step = 1;

  while (step <= blockDim.x) {
    step *= 2;
    unsigned int halfStep = step / 2;
    float angle = direction * 2 * CUDART_PI_F / step;

    unsigned int i = threadIdx.x / halfStep * step;
    unsigned int j = threadIdx.x % halfStep + i;

    Complex delta = {cosf(angle * j), sinf(angle * j)};

    Complex u = signal[j];
    Complex v = multiplyComplex(signal[j + halfStep], delta);

    signal[j] = sumComplex(u, v);
    signal[j + halfStep] = subComplex(u, v);

    __syncthreads();
  }
}



__device__ void reduceMax(Complex* signal)
{
  unsigned int i = blockDim.x / 2;

  while (i != 0) {
    if (threadIdx.x <= i) {
      signal[threadIdx.x].x = fmaxf(signal[threadIdx.x].x, signal[threadIdx.x + i].x);
    }

    __syncthreads();

    i /= 2;
  }
}



__global__ void kernel
(
  float* c,
  unsigned long long offset,
  unsigned int signalSize,
  unsigned int bitCount
)
{
  __shared__ Complex transitionMatrix[BASE];

  getTransitionMatrix(transitionMatrix);

  __syncthreads();

  extern __shared__ Complex signal[];

  getSignal(signal, transitionMatrix, offset);

  __syncthreads();

  myFFT(signal, bitCount);

  multiplyConj(signal);

  __syncthreads();

  myFFT(signal, bitCount, 1);

  signal[threadIdx.x].x = absComplex(signal[threadIdx.x + 1]) / (blockDim.x * 2);

  reduceMax(signal);

  c[blockIdx.x] = signal[0].x;
}


unsigned int bitCount_(unsigned int n) {
  unsigned int counter = 0;
  while (n != 1) {
    n /= 2;
    counter++;
  }
  return counter;
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
  unsigned long long batch,
  unsigned int signalSize,
  unsigned int bitCount
  )
{
  unsigned long long batch_size = batch * sizeof(float);

  float *host_c = (float*)malloc(batch_size);
  float host_akf;

  float *dev_c;

  cudaMalloc(&dev_c, batch_size);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  kernel<<< batch, signalSize / 2, signalSize * sizeof(Complex) >>>(dev_c, offset, signalSize, bitCount);
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
  unsigned long long numCombinations = getNumCombinations();
  size_t size = numCombinations * sizeof(float);
  unsigned int signalSize = upperPowerOfTwo(N) * 2;
  unsigned int bitCount = bitCount_(signalSize);

  if (size <= 0) {
    cout << "result array size error" << endl;
    return 1;
  }

  unsigned long long numBatches = ceil((double) size / BATCH_SIZE);

  printf("BATCH COUNT: %lld\n", numBatches);

  unsigned long long startFrom = 0;

  for (unsigned long long i = startFrom; i < numBatches; i++) {
    unsigned long long offset = i * BATCH;

    long long compare = (numCombinations - (offset + BATCH));

    unsigned long long batch = compare < 0 ? numCombinations - offset : BATCH;

    printf("\n --- BATCH %lld --- \n\n", i);

    unsigned long long batchResult = start_kernel(offset, batch, signalSize, bitCount);
  }
}