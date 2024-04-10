#include <stdint.h>
#include <stdio.h>
#include "math_constants.h"

/**
 * Вычисление матрицы перехода
*/

__device__ void getTransitionMatrix(
  float2* transition_matrix,
  uint32_t signal_size,
  uint32_t base
)
{
  for (int i = 0; i < (int) ceilf((float) base / (float) signal_size); i++) {
    int idx = threadIdx.x + i * signal_size;

    if (idx > base - 1) { break; }

    float rad = 2 * CUDART_PI * idx / base;

    transition_matrix[idx].x = cosf(rad);
    transition_matrix[idx].y = sinf(rad);
  }
}


/**
 * Генерация сигнала
*/

__device__ void getSignal(
  float2* signal,
  float2* transition_matrix,
  uint64_t offset,
  uint32_t signal_size,
  uint32_t base
)
{
  if (threadIdx.x >= signal_size) { return; }

  uint64_t signal_part = blockIdx.x + offset;

  for (int i = 0; i < threadIdx.x && threadIdx.x < signal_size; i++) {
    if (signal_part == 0) { break; }
    signal_part /= base;
  }

  uint32_t t_idx = signal_part % base;

  signal[threadIdx.x].x = transition_matrix[t_idx].x;
  signal[threadIdx.x].y = transition_matrix[t_idx].y;
}


/**
 * Поиск максимального значения
*/

__device__ void reduceMax(
  float2* signal,
  uint32_t signal_size
)
{
  uint32_t i = blockDim.x / 2;

  while (i != 0) {
    if (threadIdx.x <= i && threadIdx.x + i < signal_size) {
      signal[threadIdx.x].x = fmaxf(signal[threadIdx.x].x, signal[threadIdx.x + i].x);
    }

    __syncthreads();

    i /= 2;
  }
}


/**
 * Поиск АКФ
*/

__device__ float findAkf(
  float2* signal,
  uint32_t signal_size
)
{
  float2 sum = {0.0, 0.0};

  for (int i = 0; i + threadIdx.x < signal_size; i++) {
    sum.x += signal[threadIdx.x + i].x * signal[i].x - signal[threadIdx.x + i].y * -signal[i].y;
    sum.y += signal[threadIdx.x + i].x * -signal[i].y + signal[threadIdx.x + i].y * signal[i].x;
  }

  return sqrtf(sum.x * sum.x + sum.y * sum.y);
}


/**
 * Поиск оптимального сигнала
 * Заполняет массив float c[] максимальными лепестками
*/

__global__ void kernel(
  float *c,
  uint64_t offset,
  uint32_t signal_size,
  uint32_t base
)
{
  extern __shared__ float2 s[];

  float2 *transition_matrix = s;

  getTransitionMatrix(transition_matrix, signal_size, base);

  __syncthreads();

  float2 *signal = (float2*)&s[base];

  getSignal(
    signal,
    transition_matrix,
    offset,
    signal_size,
    base
  );

  __syncthreads();

  float akf = 0;

  if (threadIdx.x != 0) {
    akf = findAkf(signal, signal_size);
  }

  __syncthreads();

  if (threadIdx.x < signal_size) {
    signal[threadIdx.x].x = akf;
  }

  __syncthreads();

  reduceMax(signal, signal_size);

  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  c[blockIdx.x] = signal[0].x;
}


/**
 * Поиск суммы элементов массива
*/

__device__ void reduceSum(float2* arr, uint32_t signal_size)
{
  uint32_t i = blockDim.x / 2;

  while (i != 0) {
    if (threadIdx.x <= i && threadIdx.x + i < signal_size) {
      arr[threadIdx.x].x += arr[threadIdx.x + i].x;
    }

    __syncthreads();

    i /= 2;
  }
}


/**
 * Поиск оптимального сигнала с заданным сдвигом Доплера
*/

__global__ void kernel_doppler(
  float *c,
  uint64_t offset,
  uint32_t signal_size,
  uint32_t base
)
{
  extern __shared__ float2 s[];

  float2 *transition_matrix = s;

  getTransitionMatrix(transition_matrix, signal_size, base);

  __syncthreads();

  float2 *signal = (float2*)&s[base];

  getSignal(signal, transition_matrix, offset, signal_size, base);

  __syncthreads();

  float akf = findAkf(signal, signal_size);

  __syncthreads();

  float2 *akf_arr = (float2*)&s[base + signal_size];

  if (threadIdx.x < signal_size) {
    signal[threadIdx.x].x = akf;
    signal[threadIdx.x].y = threadIdx.x;

    akf_arr[threadIdx.x].x = akf;
    akf_arr[threadIdx.x].y = threadIdx.x;
  }

  __syncthreads();

  reduceMax(signal, signal_size);

  __syncthreads();

  if (threadIdx.x == 0) {
    akf_arr[(int) signal[0].y].x = 0;
  }
  
  __syncthreads();

  reduceSum(akf_arr, signal_size);

  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  c[blockIdx.x] = signal[0].x - akf_arr[0].x;
}
