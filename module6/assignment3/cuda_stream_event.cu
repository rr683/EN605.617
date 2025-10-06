// Based on stream_example.cu. Use streams for async ops, events for timing.

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define sizeOfArray 1024*1024*100

// CUDA kernel for vector addition
__global__ void vectorAddition(int *dev_a, int *dev_b, int *dev_result) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < sizeOfArray) {
        dev_result[threadId] = dev_a[threadId] + dev_b[threadId];
    }
}

// main

int main(int argc, char **argv) {
    // total threads and block size
    int totalThreads = sizeOfArray;
    int blockSize = 1024;
    if (argc > 1) totalThreads = atoi(argv[1]);
    if (argc > 2) blockSize = atoi(argv[2]);
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // Host arrays
    int *host_a, *host_b, *host_result;

    // Device arrays
    int *dev_a, *dev_b, *dev_result;

    // Allocate pinned host memory for faster transfers
    cudaHostAlloc((void**)&host_a, sizeOfArray * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, sizeOfArray * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_result, sizeOfArray * sizeof(int),
                  cudaHostAllocDefault);

    // Allocate device memory
    cudaMalloc((void**)&dev_a, sizeOfArray * sizeof(int));
    cudaMalloc((void**)&dev_b, sizeOfArray * sizeof(int));
    cudaMalloc((void**)&dev_result, sizeOfArray * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < sizeOfArray; i++) {
        host_a[i] = rand() % 10;
        host_b[i] = rand() % 10;
    }

    // Create stream and events
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, stream);

    // Async copies to device
    cudaMemcpyAsync(dev_a, host_a, sizeOfArray * sizeof(int),
                     cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, host_b, sizeOfArray * sizeof(int),
                     cudaMemcpyHostToDevice, stream);

    // Launch kernel in stream
    vectorAddition<<<numBlocks, blockSize, 0, stream>>>(dev_a, dev_b,
                                                         dev_result);

    // Async copy back to host
    cudaMemcpyAsync(host_result, dev_result, sizeOfArray * sizeof(int),
                     cudaMemcpyDeviceToHost, stream);

    // Record stop event and synchronize
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %3.3f ms\n", elapsedTime);
    printf("\n Size of Array : %d \n", sizeOfArray);


    // Print sample results
    for (int i = 0; i < 9; i++) {
        printf("Result[%d]: %d\n", i, host_result[i]);
    }

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_result);
    //cudaStreamDestroy(stream);
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);

    return;
}