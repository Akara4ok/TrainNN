//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaHelper.cuh"

void CudaHelper::calculateLinearThreadNum(int& threads_x, int& blocks_x, int size) {
    threads_x = (size >= THREADS_PER_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : size;
    blocks_x = (int) ceil(1.0 * size / threads_x);
}

void
CudaHelper::calculateBlockThreadNum(int& threads_x, int& threads_y, int& blocks_x, int& blocks_y, int height,
                                    int width) {
    threads_x = (width >= THREAD_PER_TWO_DIM_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : width;
    blocks_x = (int) ceil(1.0 * width / threads_x);
    threads_y = (height >= THREAD_PER_TWO_DIM_BLOCK) ? THREAD_PER_TWO_DIM_BLOCK : height;
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

void CudaHelper::copyFromGpuToGpu(float* src, float* dest, int size) {
    int bytes = size * sizeof(float);
    cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice);
}