//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

// Operation: Addition
void addCPU(float *a, float *b, float *c, int d) {
    for (int i = 0; i < d; i++) {
        c[i] = a[i] + b[i] + 1;
    }
}

// Kernel: following format from lecture on basic vector addition
__global__ void addGPU(float *a, float *b, float *c, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        c[i] = a[i] + b[i] + 1;
    }
}

// with branching operation
void addBranchCPU(float *a, float *b, float *c, int d) {
    for (int i = 0; i < d; i++) {
        if (a[i] > 1 && b[i] > 1) {
            c[i] = a[i] + b[i] + 1;
        } else {
            c[i] = 0;
        }
    }
}

// Kernel: following format from lecture on basic vector addition but with branching
__global__ void addBranchGPU(float *a, float *b, float *c, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        if (a[i] > 1 && b[i] > 1) {
            c[i] = a[i] + b[i] + 1;
        } else {
            c[i] = 0;
        }
    }
}

int main(int argc, char** argv) {
    // Simple arg parsing (from assignment.c and examples)
    int totalThreads = (argc >= 2) ? atoi(argv[1]) : N;
    int blockSize = (argc >= 3) ? atoi(argv[2]) : THREADS_PER_BLOCK;
    int numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        numBlocks++;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Adjusted total threads to %d\n", totalThreads);
    }

    // Allocate host memory 
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c_cpu = (float*)malloc(N * sizeof(float));
    float *h_c_gpu = (float*)malloc(N * sizeof(float));

    // Initialize data simply (random values)
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 2;  // Small random numbers
        h_b[i] = rand() % 2;
    }

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy to GPU 
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // timing CPU no branching
    clock_t start_cpu = clock();
    addCPU(h_a, h_b, h_c_cpu, N);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;  // ms
    printf("CPU no branching time: %f ms\n", cpu_time);

    // timing GPU no branching
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    addGPU<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    printf("GPU no branching time: %f ms\n", gpu_time);

    // Copy back and check
    cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("CPU without branching: c[0] = %f\n", h_c_cpu[0]);
    //printf("GPU without branching: c[0] = %f\n", h_c_gpu[0]);

    // Repeat for branching versions 
    clock_t start_cpu_branch = clock();
    addBranchCPU(h_a, h_b, h_c_cpu, N);
    clock_t end_cpu_branch = clock();
    double cpu_time_branch = ((double)(end_cpu_branch - start_cpu_branch)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU branching time: %f ms\n", cpu_time_branch);

    cudaEvent_t start_gpu_branch, stop_gpu_branch;
    cudaEventCreate(&start_gpu_branch);
    cudaEventCreate(&stop_gpu_branch);
    cudaEventRecord(start_gpu_branch);
    addBranchGPU<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop_gpu_branch);
    cudaEventSynchronize(stop_gpu_branch);
    float gpu_time_branch;
    cudaEventElapsedTime(&gpu_time_branch, start_gpu_branch, stop_gpu_branch);
    printf("GPU branching time: %f ms\n", gpu_time_branch);

    cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("CPU with branching: c[0] = %f\n", h_c_cpu[0]);
    //printf("GPU with branching: c[0] = %f\n", h_c_gpu[0]);

    // Clean up 
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}