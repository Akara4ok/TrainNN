//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaFunctions.cuh"

namespace GPU {
    __global__ void zeroInit(float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            data[row * width + col] = 0;
        }
    }

    __global__ void multiply(float* data, int height, int width, float value) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            data[row * width + col] = data[row * width + col] * value;
        }
    }

    __global__ void sum(float* result, const float* data, int height, int width, int axis) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            switch (axis) {
                case -1:
                    atomicAdd(result, data[row * width + col]);
                    break;
                case 0:
                    atomicAdd(result + row, data[row * width + col]);
                    break;
                case 1:
                    atomicAdd(result + col, data[row * width + col]);
                    break;
                default:
                    break;
            }
        }
    }

    __global__ void exp(float* result, const float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = expf(data[row * width + col]);
        }
    }

    __global__ void log(float* result, const float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = logf(data[row * width + col]);
        }
    }

    __global__ void argmax(float* result, const float* data, int height, int width, int axis) {
        const unsigned int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (axis == 0) {
            if (threadId < height) {
                data += threadId * width;
                float maxValue = -1;
                float maxInd = -1;
                for (int i = 0; i < width; i++) {
                    if (data[i] > maxValue) {
                        maxValue = data[i];
                        maxInd = static_cast<float>(i);
                    }
                }
                result[threadId] = maxInd;
            }
        } else if (axis == 1) {
            if (threadId < width) {
                float maxValue = -1;
                float maxInd = -1;
                for (int i = 0; i < height; i++) {
                    if (data[i * width + threadId] > maxValue) {
                        maxValue = data[i * width + threadId];
                        maxInd = static_cast<float>(i);
                    }
                }
                result[threadId] = maxInd;
            }
        }
    }

    __global__ void reciprocal(float* result, const float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = 1 / data[row * width + col];
        }
    }

    __global__ void clip(float* result, const float* data, int height, int width,
                         float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = data[row * width + col];
            if (data[row * width + col] < minBound) {
                result[row * width + col] = minValueToSet;
            }
            if (data[row * width + col] > maxBound) {
                result[row * width + col] = maxValueToSet;
            }
        }
    }

    __global__ void transpose(float* result, const float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[col * height + row] = data[row * width + col];
        }
    }

    __global__ void
    sum(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs, int heightRhs,
        int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] + rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    subtract(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs, int heightRhs,
             int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] - rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    elementWiseMultiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
                        int heightRhs, int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] * rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    elementWiseDivide(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
                      int heightRhs, int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] / rhsData[rowRhs * widthRhs + colRhs];
        }
    }

#ifdef CUDA_STANDARD_MULT
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthRhs)) {
            result[row * widthRhs + col] = 0;
            for (int i = 0; i < widthLhs; i++) {
                result[row * widthRhs + col] += lhsData[row * widthLhs + i] * rhsData[i * widthRhs + col];
            }
        }
    }
#endif
#ifdef CUDA_SHARED_MULT
    const int TILE_MAX = 32;
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        __shared__ float A_tile[TILE_MAX][TILE_MAX];
        __shared__ float B_tile[TILE_MAX][TILE_MAX];
        float acc = 0;
        const int tiles = (TILE_MAX + widthLhs - 1) / TILE_MAX;
        for (int tile = 0; tile < tiles; tile++){
            A_tile[threadIdx.y][threadIdx.x] = 0;
            B_tile[threadIdx.y][threadIdx.x] = 0;
            const unsigned int col_j = (tile * TILE_MAX) + threadIdx.x;
            const unsigned int row_j = (tile * TILE_MAX) + threadIdx.y;
            if (col_j < widthLhs && row < heightLhs)
                A_tile[threadIdx.y][threadIdx.x] = lhsData[row * widthLhs + col_j];
            if(row_j < widthLhs && col < widthRhs)
                B_tile[threadIdx.y][threadIdx.x] = rhsData[row_j * widthRhs + col];
            __syncthreads();
//            printf("%i\n", threadIdx.x);
            for (int i = 0; i < TILE_MAX; i++){
                acc += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
            }
            __syncthreads();
        }
        if ((row < heightLhs) && (col < widthRhs)) {
            result[row * widthRhs + col] = acc;
        }
    }
#endif
}