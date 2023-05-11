//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaHelper.cuh"

void CudaHelper::calculateLinearThreadNum(int& threadsX, int& blocksX, int size) {
    threadsX = (size >= THREADS_PER_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : size;
    blocksX = (int) ceil(1.0 * size / threadsX);
}

void
CudaHelper::calculateBlockThreadNum(int& threadsX, int& threadsY, int& blocksX, int& blocksY, int height,
                                    int width) {
    threadsX = (width >= THREAD_PER_TWO_DIM_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : width;
    blocksX = ceil(1.0 * width / threadsX);
    threadsY = (height >= THREAD_PER_TWO_DIM_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : height;
    blocksY = ceil(1.0 * height / threadsY);
}

void CudaHelper::allocateGpuMemory(float** data, int size) {
    int bytes = static_cast<int>(size * sizeof(float));
    cudaMalloc(data, bytes);
}

void CudaHelper::deleteGpuMemory(float* data) {
    cudaFree(data);
}

void CudaHelper::copyFromCpuToGpu(float* cpuData, float* gpuData, int size) {
    int bytes = static_cast<int>(size * sizeof(float));
    cudaMemcpy(gpuData, cpuData, bytes, cudaMemcpyHostToDevice);
}

void CudaHelper::copyFromGpuToCpu(float* gpuData, float* cpuData, int size) {
    int bytes = static_cast<int>(size * sizeof(float));
    cudaMemcpy(cpuData, gpuData, bytes, cudaMemcpyDeviceToHost);
}

void CudaHelper::copyFromGpuToGpu(float* src, float* dest, int size) {
    int bytes = static_cast<int>(size * sizeof(float));
    cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice);
}