#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#define SIZE 1024  
// Generate random floats using cuRAND
void generateRandomData(curandGenerator_t gen, float *d_data, int size) {
    curandGenerateUniform(gen, d_data, size);
}

// Perform matrix multiplication using cuBLAS
void matrixMultiply(cublasHandle_t cublas, float *d_a, float *d_b, float *d_c) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE, SIZE,
                &alpha, d_a, SIZE, d_b, SIZE, &beta, d_c, SIZE);
}

int main() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, SIZE * SIZE * sizeof(float));
    cudaMalloc(&d_b, SIZE * SIZE * sizeof(float));
    cudaMalloc(&d_c, SIZE * SIZE * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // cuRAND: Generate random matrices
    generateRandomData(gen, d_a, SIZE * SIZE);
    generateRandomData(gen, d_b, SIZE * SIZE);

    // cuBLAS: Multiply 
    matrixMultiply(cublas, d_a, d_b, d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Time: " << time << " ms" << std::endl;

    // Copy back sample and print
    float *h_c = new float[SIZE * SIZE];
    cudaMemcpy(h_c, d_c, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sample result: " << h_c[0] << std::endl;

    // Print matrix size and more samples
    std::cout << "Matrix size: " << SIZE << "x" << SIZE << std::endl;
    std::cout << "More samples: " << h_c[1] << ", " << h_c[2] << ", " << h_c[3] << std::endl;

    // Basic stats
    float min = h_c[0], max = h_c[0], sum = 0.0f;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (h_c[i] < min) min = h_c[i];
        if (h_c[i] > max) max = h_c[i];
        sum += h_c[i];
    }
    float avg = sum / (SIZE * SIZE);
    std::cout << "Min: " << min << ", Max: " << max << ", Avg: " << avg << std::endl;

    // Cleanup
    curandDestroyGenerator(gen);
    cublasDestroy(cublas);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_c;
    return 0;
}