#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <float.h>

#define W 1024 * 16 
#define H 1024 * 16

void fillArr(int* arr, unsigned int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10;
    }
}

void fillArrZero(int* arr, unsigned int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 0;
    }
}

void printMatrix(const int* arr, const unsigned int n, const unsigned int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("[%d] ", arr[W * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void subKernel(const int* a, const int* b, int* c) {
    int i = W * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] - b[i];
}

void parallel(const int* a, const int* b, int* c, unsigned int n, unsigned int m, int threadsNum) {
    unsigned int size = n * m;
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    dim3 threadsPerBlock(threadsNum);
    dim3 blocksPerGrid(size / threadsPerBlock.x);

    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    subKernel <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void sequential(const int* a, const int* b, int* c, unsigned int n, unsigned int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            unsigned int index = n * i + j;
            c[index] = a[index] - b[index];
        }
    }
}

int main() {
    srand(time(NULL));
    const unsigned int size = W * H;
    int* a = new int[size];
    int* b = new int[size];
    int* c = new int[size];

    fillArr(a, size);
    fillArr(b, size);
    fillArrZero(c, size);

    unsigned printW = 5, printH = 5;
    printf("Matrix size = %d x %d\n", W, H);
    printf("Matrix part (%d x %d):\n", printW, printH);
    printf("a = \n");
    printMatrix(a, printW, printH);
    printf("b = \n");
    printMatrix(b, printW, printH);

    clock_t start, end;
    start = CLOCKS_PER_SEC;
    sequential(a, b, c, W, H);
    end = clock();
    printf("result (sequential) = \n");
    printMatrix(c, printW, printH);
    printf("time (sequential) = %.2f ms\n", ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0);
    printf("\n");

    double minTime = DBL_MAX;
    int bestThreadsNum;

    for (int i = 0; i < 10; i++) {
        int threadsNum = (int)pow(2, i);
        fillArrZero(c, W);
        start = clock();
        parallel(a, b, c, W, H, threadsNum);
        end = clock();

        double diff = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

        printf("threads number = %d\n", threadsNum);
        printf("time = %.2f ms\n", diff);

        if (diff < minTime) {
            minTime = diff;
            bestThreadsNum = threadsNum;
        }
    }

    fillArrZero(c, W);
    parallel(a, b, c, W, H, bestThreadsNum);
    printf("\n");
    printf("result (parallel) =\n");
    printMatrix(c, printW, printH);
    printf("best time: %f, threads: %d (parallel)\n", minTime, bestThreadsNum);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}

/*
threads number = 1
time = 1523.00 ms
threads number = 2
time = 1000.00 ms
threads number = 4
time = 796.00 ms
threads number = 8
time = 698.00 ms
threads number = 16
time = 643.00 ms
threads number = 32
time = 618.00 ms
threads number = 64
time = 623.00 ms
threads number = 128
time = 631.00 ms
threads number = 256
time = 630.00 ms
threads number = 512
time = 631.00 ms

best time: 618.000000, threads: 32 (parallel)

Наилучшее время при конфигурации сетки с количеством нитей равным 32.
*/