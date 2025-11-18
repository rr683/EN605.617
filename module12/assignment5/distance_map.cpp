#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#define WIDTH 16
#define HEIGHT 16

int main(int argc, char **argv) {
    int width = (argc > 1) ? atoi(argv[1]) : WIDTH;
    int height = (argc > 2) ? atoi(argv[2]) : HEIGHT;

    // Setup grid
    int *grid = new int[width * height];
    for (int i = 0; i < width * height; i++) grid[i] = 0;
    grid[5 * width + 5] = 1;

    // OpenCL setup with error checking
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { std::cerr << "Platform error\n"; return 1; }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "No GPU found, trying CPU\n";
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) { std::cerr << "No device found\n"; return 1; }
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { std::cerr << "Context error\n"; return 1; }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) { std::cerr << "Queue error\n"; return 1; }

    // Buffers
    cl_mem gridBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    sizeof(int) * width * height, grid, &err);
    if (err != CL_SUCCESS) { std::cerr << "Grid buffer error\n"; return 1; }

    cl_mem distBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                    sizeof(float) * width * height, NULL, &err);
    if (err != CL_SUCCESS) { std::cerr << "Dist buffer error\n"; return 1; }

    // Load kernel
    std::ifstream file("distance_map.cl");
    if (!file.is_open()) { std::cerr << "Kernel file not found\n"; return 1; }
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char *srcPtr = src.c_str();
    size_t srcSize = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcSize, &err);
    if (err != CL_SUCCESS) { std::cerr << "Program error\n"; return 1; }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[1024];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        std::cerr << "Build error:\n" << log << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "computeDistanceMap", &err);
    if (err != CL_SUCCESS) { std::cerr << "Kernel error\n"; return 1; }

    // Set args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &gridBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &distBuf);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);

    // Execute
    size_t globalSize[2] = {(size_t)width, (size_t)height};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "Execution error: " << err << std::endl; return 1; }

    // Read results
    float *distMap = new float[width * height];
    err = clEnqueueReadBuffer(queue, distBuf, CL_TRUE, 0, sizeof(float) * width * height, distMap, 0, NULL, NULL);
    if (err != CL_SUCCESS) { std::cerr << "Read error\n"; return 1; }

    // Print map
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << distMap[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] grid; delete[] distMap;
    clReleaseMemObject(gridBuf); clReleaseMemObject(distBuf);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return 0;
}