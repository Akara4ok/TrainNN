//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaHelper.cuh"

void
CudaHelper::calculateThreadNum(int& threads_x, int& threads_y, int& blocks_x, int& blocks_y, int height, int width) {
    threads_x = (width >= 32) ? THREAD_PER_TWO_DIM_BLOCK : width;
    blocks_x = (int) ceil(1.0 * width / threads_x);
    threads_y = (height >= 32) ? THREAD_PER_TWO_DIM_BLOCK : (int) ceil(height);
    blocks_y = (int) ceil(1.0 * height / threads_y);
}

void CudaHelper::allocateGpuMemory(float** data, int size) {
    int bytes = size * sizeof(float);
    cudaMalloc(data, bytes);
}

void CudaHelper::deleteGpuMemory(float* data) {
    cudaFree(data);
}

void CudaHelper::copyFromCpuToGpu(float* cpuData, float* gpuData, int size) {
    int bytes = size * sizeof(float);
    cudaMemcpy(gpuData, cpuData, bytes, cudaMemcpyHostToDevice);
}

void CudaHelper::copyFromGpuToCpu(float* gpuData, float* cpuData, int size) {
    int bytes = size * sizeof(float);
    cudaMemcpy(cpuData, gpuData, bytes, cudaMemcpyDeviceToHost);
}