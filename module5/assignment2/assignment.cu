#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  // Array size, adjustable
#define A 2.0f  // Scalar for SAXPY

// Host memory SAXPY 
void saxpy_host(float *x, float *y, int n, float a) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Global memory kernel
__global__ void saxpy_global(float *x, float *y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Register memory kernel
__global__ void saxpy_register(float *x, float *y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];  // Register
        float yi = y[i];  // Register
        yi = a * xi + yi;
        y[i] = yi;
    }
}

// Constant memory
__constant__ float const_a;
__global__ void saxpy_constant(float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = const_a * x[i] + y[i];
    }
}

// Shared memory kernel
__global__ void saxpy_shared(float *x, float *y, int n, float a) {
    __shared__ float shared_x[256];  // Block size max
    __shared__ float shared_y[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    if (i < n) {
        shared_x[local_i] = x[i];
        shared_y[local_i] = y[i];
        __syncthreads();
        shared_y[local_i] = a * shared_x[local_i] + shared_y[local_i];
        y[i] = shared_y[local_i];
    }
}

void execute_host_functions(float *x, float *y, int n) {
    printf("Host memory SAXPY:\n");
    saxpy_host(x, y, n, A);
    for (int i = 0; i < 5; i++) printf("y[%d]=%f\n", i, y[i]);
}

void execute_gpu_functions(int total_threads, int block_size) {
    int num_blocks = (total_threads + block_size - 1) / block_size;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Init data
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Global memory timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy_global<<<num_blocks, block_size>>>(d_x, d_y, N, A);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_global;
    cudaEventElapsedTime(&ms_global, start, stop);
    printf("Global memory: %.3f ms\n", ms_global);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Register memory timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy_register<<<num_blocks, block_size>>>(d_x, d_y, N, A);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_register;
    cudaEventElapsedTime(&ms_register, start, stop);
    printf("Register memory: %.3f ms\n", ms_register);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Constant memory timing
    float host_a = A;
    cudaMemcpyToSymbol(const_a, &host_a, sizeof(float));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy_constant<<<num_blocks, block_size>>>(d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_constant;
    cudaEventElapsedTime(&ms_constant, start, stop);
    printf("Constant memory: %.3f ms\n", ms_constant);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Shared memory timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy_shared<<<num_blocks, block_size>>>(d_x, d_y, N, A);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared;
    cudaEventElapsedTime(&ms_shared, start, stop);
    printf("Shared memory: %.3f ms\n", ms_shared);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}

int main(int argc, char** argv) {
    int total_threads = 64;  // Min 64
    int block_size = 64;
    if (argc > 1) total_threads = atoi(argv[1]);
    if (argc > 2) block_size = atoi(argv[2]);
    int num_blocks = (total_threads + block_size - 1) / block_size;

    printf("Threads: %d, Blocks: %d, Grid: %d\n",
           total_threads, block_size, num_blocks);

    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    execute_host_functions(x, y, N);
    execute_gpu_functions(total_threads, block_size);

    free(x);
    free(y);
    return 0;
}