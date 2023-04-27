//
// Created by vlad on 4/23/23.
//

#include <cuda.h>
#include <curand.h>
#include "Matrix/Matrix.h"
#include "Cuda/CudaHelper.cuh"
#ifdef CUDA_STANDARD
#include "Cuda/CudaKernelFunctions.cuh"
#endif
#include "Matrix/Calculation/GpuMatrixCalculation.cuh"

Matrix GpuMatrixCalculation::sum(const Matrix& matrix, int axis) {
    return {};
}

Matrix GpuMatrixCalculation::multiply(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::exp(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::exp_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::log(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::log_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::transpose(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::transpose_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::elementWiseDivide(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix
GpuMatrixCalculation::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                           float maxValueToSet) {
    return {};
}

void GpuMatrixCalculation::clip_inline(Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {

}

Matrix GpuMatrixCalculation::sum(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::subtract(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::reciprocal(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::reciprocal_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::argmax(const Matrix& matrix, int axis) {
    return {};
}

void GpuMatrixCalculation::randomInit(Matrix& matrix, int w) {
    float* gpuData = matrix.getGpuData();
    if(gpuData != nullptr){
        CudaHelper::deleteGpuMemory(gpuData);
    }

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());

    curandGenerateNormal(gen, gpuData,
                         matrix.getWidth() * matrix.getHeight(),
                         0.0f, 1.0f);

    curandDestroyGenerator(gen);

    int threadsNumX, blocksNumX, threadsNumY,  blocksNumY;
    CudaHelper::calculateThreadNum(threadsNumX, threadsNumY,
                                   blocksNumX, blocksNumY,
                                   matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY);
    const dim3 blocks(blocksNumX, blocksNumY);
    GPU::scale<<<threads, blocks>>>(gpuData,
                                         matrix.getHeight(),
                                         matrix.getWidth(), sqrt(w));
    matrix.setGpuData(gpuData);
}

void GpuMatrixCalculation::zeroInit(Matrix& matrix) {
    float* gpuData = matrix.getGpuData();
    if(gpuData != nullptr){
        CudaHelper::deleteGpuMemory(gpuData);
    }
    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY,  blocksNumY;
    CudaHelper::calculateThreadNum(threadsNumX, threadsNumY,
                                   blocksNumX, blocksNumY,
                                   matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY);
    const dim3 blocks(blocksNumX, blocksNumY);
    GPU::zeroInit<<<threads, blocks>>>(gpuData,
                                       matrix.getHeight(), matrix.getWidth());
    matrix.setGpuData(gpuData);
}