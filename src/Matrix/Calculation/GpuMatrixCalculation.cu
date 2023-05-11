//
// Created by vlad on 4/23/23.
//

#include <curand.h>
#include "Matrix/Matrix.h"
#include "Cuda/CudaHelper.cuh"
#include "Cuda/CudaFunctions.cuh"
#include "Matrix/Calculation/GpuMatrixCalculation.cuh"

Matrix GpuMatrixCalculation::sum(const Matrix& matrix, int axis) {
    Matrix result(axis == 0 ? matrix.getHeight() : 1,
                  axis == 1 ? matrix.getWidth() : 1,
                  Provider::GPU);

    result.zeroInit();
    float* gpuData = result.getGpuData();

    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::sum<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth(),
            axis);

    return result;
}

Matrix GpuMatrixCalculation::multiply(const Matrix& lhs, const Matrix& rhs) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, lhs.getHeight() * rhs.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
#ifdef CUDA_STANDARD_MULT
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        lhs.getHeight(), rhs.getWidth());
#endif
#if defined(CUDA_COALESCING_MULT) || defined(CUDA_SHARED_BLOCK_MULT)
    threadsNumX = CudaHelper::THREAD_PER_TWO_DIM_BLOCK * CudaHelper::THREAD_PER_TWO_DIM_BLOCK;
    threadsNumY = 1;
    blocksNumX = (rhs.getWidth() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
    blocksNumY = (lhs.getHeight() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
#endif
#if defined(CUDA_SHARED1D_MULT)
    threadsNumX = GPU::BHL * GPU::BWR / GPU::TWR;
    threadsNumY = 1;
    blocksNumX = (rhs.getWidth() - 1) / GPU::BWR + 1;
    blocksNumY = (lhs.getHeight() - 1) / GPU::BHL + 1;
#endif
#if defined(CUDA_SHARED2D_MULT)
    threadsNumX = GPU::BHL * GPU::BWR / (GPU::THL * GPU::TWR);
    threadsNumY = 1;
    blocksNumX = (rhs.getWidth() - 1) / GPU::BWR + 1;
    blocksNumY = (lhs.getHeight() - 1) / GPU::BHL + 1;
#endif

    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);

    GPU::multiply<<<blocks, threads>>>(gpuData,
            lhs.getGpuData(),
            rhs.getGpuData(),
            lhs.getHeight(),
            lhs.getWidth(),
            rhs.getWidth());

    return {gpuData, lhs.getHeight(), rhs.getWidth(), Provider::GPU};
}

Matrix GpuMatrixCalculation::exp(const Matrix& matrix) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::exp<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());

    return {gpuData, matrix.getHeight(), matrix.getWidth(), Provider::GPU};
}

void GpuMatrixCalculation::expInline(Matrix& matrix) {
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::exp<<<blocks, threads>>>(matrix.getGpuData(),
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());
}

Matrix GpuMatrixCalculation::log(const Matrix& matrix) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::log<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());

    return {gpuData, matrix.getHeight(), matrix.getWidth(), Provider::GPU};
}

void GpuMatrixCalculation::logInline(Matrix& matrix) {
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::log<<<blocks, threads>>>(matrix.getGpuData(),
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());
}

Matrix GpuMatrixCalculation::transpose(const Matrix& matrix) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
#ifdef CUDA_STANDARD_TRANSPOSE
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
#endif
#if defined(CUDA_SHARED_TRANSPOSE) || defined(CUDA_NO_BANK_TRANSPOSE)
    threadsNumX = CudaHelper::THREAD_PER_TWO_DIM_BLOCK;
    threadsNumY = GPU::BWL;
    blocksNumX = (matrix.getWidth() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
    blocksNumY = (matrix.getHeight() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
#endif
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::transpose<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());

    return {gpuData, matrix.getWidth(), matrix.getHeight(), Provider::GPU};
}

void GpuMatrixCalculation::transposeInline(Matrix& matrix) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
#ifdef CUDA_STANDARD_TRANSPOSE
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
#endif
#if defined(CUDA_SHARED_TRANSPOSE) || defined(CUDA_NO_BANK_TRANSPOSE)
    threadsNumX = CudaHelper::THREAD_PER_TWO_DIM_BLOCK;
    threadsNumY = GPU::BWL;
    blocksNumX = (matrix.getWidth() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
    blocksNumY = (matrix.getHeight() - 1) / CudaHelper::THREAD_PER_TWO_DIM_BLOCK + 1;
#endif
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::transpose<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());

    CudaHelper::deleteGpuMemory(matrix.getGpuData());
    matrix.setNewGpuDataWithSize(gpuData, matrix.getWidth(), matrix.getHeight());
}

Matrix GpuMatrixCalculation::elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, lhs.getHeight() * lhs.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        lhs.getHeight(), lhs.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::elementWiseMultiply<<<blocks, threads>>>(gpuData,
            lhs.getGpuData(),
            rhs.getGpuData(),
            lhs.getHeight(),
            lhs.getWidth(),
            rhs.getHeight(),
            rhs.getWidth());

    return {gpuData, lhs.getHeight(), lhs.getWidth(), Provider::GPU};
}

Matrix GpuMatrixCalculation::elementWiseDivide(const Matrix& lhs, const Matrix& rhs) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, lhs.getHeight() * lhs.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        lhs.getHeight(), lhs.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::elementWiseDivide<<<blocks, threads>>>(gpuData,
            lhs.getGpuData(),
            rhs.getGpuData(),
            lhs.getHeight(),
            lhs.getWidth(),
            rhs.getHeight(),
            rhs.getWidth());

    return {gpuData, lhs.getHeight(), lhs.getWidth(), Provider::GPU};
}

Matrix
GpuMatrixCalculation::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                           float maxValueToSet) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::clip<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth(),
            minBound, maxBound,
            minValueToSet, maxValueToSet);

    return {gpuData, matrix.getHeight(), matrix.getWidth(), Provider::GPU};
}

void GpuMatrixCalculation::clipInline(Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                                      float maxValueToSet) {
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::clip<<<blocks, threads>>>(matrix.getGpuData(),
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth(),
            minBound, maxBound,
            minValueToSet, maxValueToSet);
}

Matrix GpuMatrixCalculation::sum(const Matrix& lhs, const Matrix& rhs) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, lhs.getHeight() * lhs.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        lhs.getHeight(), lhs.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::sum<<<blocks, threads>>>(gpuData,
            lhs.getGpuData(),
            rhs.getGpuData(),
            lhs.getHeight(),
            lhs.getWidth(),
            rhs.getHeight(),
            rhs.getWidth());

    return {gpuData, lhs.getHeight(), lhs.getWidth(), Provider::GPU};
}

Matrix GpuMatrixCalculation::subtract(const Matrix& lhs, const Matrix& rhs) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, lhs.getHeight() * lhs.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        lhs.getHeight(), lhs.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::subtract<<<blocks, threads>>>(gpuData,
            lhs.getGpuData(),
            rhs.getGpuData(),
            lhs.getHeight(),
            lhs.getWidth(),
            rhs.getHeight(),
            rhs.getWidth());

    return {gpuData, lhs.getHeight(), lhs.getWidth(), Provider::GPU};
}

Matrix GpuMatrixCalculation::reciprocal(const Matrix& matrix) {
    float* gpuData;

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::reciprocal<<<blocks, threads>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());

    return {gpuData, matrix.getHeight(), matrix.getWidth(), Provider::GPU};
}

void GpuMatrixCalculation::reciprocalInline(Matrix& matrix) {
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY, 1);
    const dim3 blocks(blocksNumX, blocksNumY, 1);
    GPU::reciprocal<<<blocks, threads>>>(matrix.getGpuData(),
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth());
}

Matrix GpuMatrixCalculation::argmax(const Matrix& matrix, int axis) {
    float* gpuData;

    int threadsNum, blocksNum;
    if (axis == 0) {
        CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight());
        CudaHelper::calculateLinearThreadNum(threadsNum, blocksNum, matrix.getHeight());
    } else {
        CudaHelper::allocateGpuMemory(&gpuData, matrix.getWidth());
        CudaHelper::calculateLinearThreadNum(threadsNum, blocksNum, matrix.getWidth());
    }
    GPU::argmax<<<blocksNum, threadsNum>>>(gpuData,
            matrix.getGpuData(),
            matrix.getHeight(),
            matrix.getWidth(),
            axis);

    return {gpuData,
            axis == 0 ? matrix.getHeight() : 1,
            axis == 1 ? matrix.getWidth() : 1,
            Provider::GPU};
}

void GpuMatrixCalculation::randomInit(Matrix& matrix, int w) {
    float* gpuData;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());

    curandGenerateNormal(gen, gpuData,
                         matrix.getWidth() * matrix.getHeight(),
                         0.0f, 1.0f);

    curandDestroyGenerator(gen);

    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY);
    const dim3 blocks(blocksNumX, blocksNumY);
    GPU::multiply<<<blocks, threads>>>(gpuData,
            matrix.getHeight(),
            matrix.getWidth(), sqrtf(2.f / static_cast<float>(w)));
    matrix.setGpuData(gpuData);
}

void GpuMatrixCalculation::zeroInit(Matrix& matrix) {
    float* gpuData;
    CudaHelper::allocateGpuMemory(&gpuData, matrix.getHeight() * matrix.getWidth());
    int threadsNumX, blocksNumX, threadsNumY, blocksNumY;
    CudaHelper::calculateBlockThreadNum(threadsNumX, threadsNumY,
                                        blocksNumX, blocksNumY,
                                        matrix.getHeight(), matrix.getWidth());
    const dim3 threads(threadsNumX, threadsNumY);
    const dim3 blocks(blocksNumX, blocksNumY);
    GPU::zeroInit<<<blocks, threads>>>(gpuData,
            matrix.getHeight(), matrix.getWidth());
    matrix.setGpuData(gpuData);
}