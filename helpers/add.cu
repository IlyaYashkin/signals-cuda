#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <chrono>

#define N 16777216

__global__ void addKernel(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* a, int* b, int* c)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaEvent_t start, stop;

    cudaSetDevice(0);
    
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blockPerGrid(N / threadsPerBlock.x);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    addKernel<<<blockPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float t;

    cudaEventElapsedTime(&t, start, stop);
    printf("gpu time: %f\n", t);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

int main()
{
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = (rand() % 10);
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        b[i] = (rand() % 10);
    }

    addWithCuda(a, b, c);
    cudaDeviceReset();

    for (int i = N - 10; i < N; i++) {
        printf("%d\n", c[i]);
    }

    // Вычислил в этом же файле на ЦПУ
    int* c_cpu = new int[N];
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        c_cpu[i] = a[i] + b[i];
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    printf("cpu time: %f", static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count()) / 1000);

    return 0;
}