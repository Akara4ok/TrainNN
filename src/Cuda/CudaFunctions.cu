//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaFunctions.cuh"
#include "stdio.h"

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
        const uint row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < heightLhs) && (col < widthRhs)) {
            result[row * widthRhs + col] = 0;
            for (int i = 0; i < widthLhs; i++) {
                result[row * widthRhs + col] += lhsData[row * widthLhs + i] * rhsData[i * widthRhs + col];
            }
        }
    }
#endif
#ifdef CUDA_COALSCING_MULT
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const int BLOCKSIZE = 32;
        const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
        const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
        if ((row < heightLhs) && (col < widthRhs)) {
            float acc = 0.0;
            for (int i = 0; i < widthLhs; i++) {
                acc += lhsData[row * widthLhs + i] * rhsData[i * widthRhs + col];
            }
            result[row * widthRhs + col] = acc;
        }
    }
#endif
#ifdef CUDA_SHAREDBLOCK_MULT
    const int BLOCKSIZE = 32;
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const uint tRow = threadIdx.x / BLOCKSIZE;
        const uint tCol = threadIdx.x % BLOCKSIZE;
        const uint row = blockIdx.y * BLOCKSIZE + tRow;
        const uint col = blockIdx.x * BLOCKSIZE + tCol;
        __shared__ float A_tile[BLOCKSIZE][BLOCKSIZE];
        __shared__ float B_tile[BLOCKSIZE][BLOCKSIZE];
        float acc = 0;
        const int tiles = (BLOCKSIZE + widthLhs - 1) / BLOCKSIZE;
        for (int tile = 0; tile < tiles; tile++){
            A_tile[tRow][tCol] = 0;
            B_tile[tRow][tCol] = 0;
            const uint col_j = (tile * BLOCKSIZE) + tCol;
            const uint row_j = (tile * BLOCKSIZE) + tRow;
            if (col_j < widthLhs && row < heightLhs)
                A_tile[tRow][tCol] = lhsData[row * widthLhs + col_j];
            if(row_j < widthLhs && col < widthRhs)
                B_tile[tRow][tCol] = rhsData[row_j * widthRhs + col];
            __syncthreads();
//            printf("%i\n", threadIdx.x);
            for (int i = 0; i < BLOCKSIZE; i++){
                acc += A_tile[tRow][i] * B_tile[i][tCol];
            }
            __syncthreads();
        }
        if ((row < heightLhs) && (col < widthRhs)) {
            result[row * widthRhs + col] = acc;
        }
    }
#endif
#ifdef CUDA_SHARED1D_MULT
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int TM = 8;

        const uint tRow = threadIdx.x / BN;
        const uint tCol = threadIdx.x % BN;
        __shared__ float A_tile[BM * BK];
        __shared__ float B_tile[BK * BN];

        const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
        const uint innerRowA = threadIdx.x / BK;
        const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
        const uint innerRowB = threadIdx.x / BN;

        float threadResults[TM] = {0.0};

        for (uint bkIdx = 0; bkIdx < widthLhs; bkIdx += BK) {
            // populate the SMEM caches
            A_tile[innerRowA * BK + innerColA] = 0;
            B_tile[innerRowB * BN + innerColB] = 0;
            if((blockIdx.y * BM + innerRowA) < heightLhs && bkIdx + innerColA < widthLhs)
                A_tile[innerRowA * BK + innerColA] = lhsData[(blockIdx.y * BM + innerRowA) * widthLhs + bkIdx + innerColA];
            if(bkIdx + innerRowB < widthLhs && innerColB + blockIdx.x * BN < widthRhs)
                B_tile[innerRowB * BN + innerColB] = rhsData[(bkIdx + innerRowB) * widthRhs + innerColB + blockIdx.x * BN];
            __syncthreads();


            // calculate per-thread results
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // we make the dotproduct loop the outside loop, which facilitates
                // reuse of the Bs entry, which we can cache in a tmp var.
                float tmpB = B_tile[dotIdx * BN + tCol];
                for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                    threadResults[resIdx] +=
                            A_tile[(tRow * TM + resIdx) * BK + dotIdx] * tmpB;
                }
            }
            __syncthreads();
        }

        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            if ((tRow * TM + resIdx + blockIdx.y * BM) < heightLhs && tCol + blockIdx.x * BN< widthRhs)
                result[(tRow * TM + resIdx + blockIdx.y * BM) * widthRhs + tCol + blockIdx.x * BN] = threadResults[resIdx];
        }
    }
#endif
#ifdef CUDA_SHARED2D_MULT
    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;

        const uint totalResultsBlocktile = BM * BN;
        const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

        const uint tRow = threadIdx.x / (BN / TN);
        const uint tCol = threadIdx.x % (BN / TN);
        __shared__ float A_tile[BM * BK];
        __shared__ float B_tile[BK * BN];

        const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
        const uint innerRowA = threadIdx.x / BK;
        const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
        const uint innerRowB = threadIdx.x / BN;

        const uint strideA = numThreadsBlocktile / BK;
        const uint strideB = numThreadsBlocktile / BN;

        float threadResults[TM * TN] = {0.0};

        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        for (uint bkIdx = 0; bkIdx < widthLhs; bkIdx += BK) {
            // populate the SMEM caches
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                A_tile[(innerRowA + loadOffset) * BK + innerColA] = 0;
                if(innerRowA + loadOffset + blockIdx.y * BM < heightLhs && bkIdx + innerColA < widthLhs)
                    A_tile[(innerRowA + loadOffset) * BK + innerColA] =
                            lhsData[(innerRowA + loadOffset + blockIdx.y * BM) * widthLhs + bkIdx + innerColA];
            }
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                B_tile[(innerRowB + loadOffset) * BN + innerColB] = 0;
                if(bkIdx + innerRowB + loadOffset < widthLhs && innerColB + blockIdx.x * BN < widthRhs)
                    B_tile[(innerRowB + loadOffset) * BN + innerColB] =
                            rhsData[(bkIdx + innerRowB + loadOffset) * widthRhs + innerColB + blockIdx.x * BN];
            }

//            lhsData += BK;     // move BK columns to right
//            rhsData += BK * widthRhs; // move BK rows down

//            A_tile[innerRowA * BK + innerColA] = 0;
//            B_tile[innerRowB * BN + innerColB] = 0;
//            if((blockIdx.y * BM + innerRowA) < heightLhs && bkIdx + innerColA < widthLhs)
//                A_tile[innerRowA * BK + innerColA] = lhsData[(blockIdx.y * BM + innerRowA) * widthLhs + bkIdx + innerColA];
//            if(bkIdx + innerRowB < widthLhs && innerColB + blockIdx.x * BN < widthRhs)
//                B_tile[innerRowB * BN + innerColB] = rhsData[(bkIdx + innerRowB) * widthRhs + innerColB + blockIdx.x * BN];
            __syncthreads();


            // calculate per-thread results
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // block into registers
                for (uint i = 0; i < TM; ++i) {
                    regM[i] = A_tile[(tRow * TM + i) * BK + dotIdx];
                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = B_tile[dotIdx * BN + tCol * TN + i];
                }
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                if(tRow * TM + resIdxM + blockIdx.y * BM < heightLhs && tCol * TN + blockIdx.x * BN + resIdxN < widthRhs)
                    result[(tRow * TM + resIdxM + blockIdx.y * BM) * widthRhs + tCol * TN + blockIdx.x * BN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
            }
        }


//        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//            if ((tRow * TM + resIdx + blockIdx.y * BM) < heightLhs && tCol + blockIdx.x * BN< widthRhs)
//                result[(tRow * TM + resIdx + blockIdx.y * BM) * widthRhs + tCol + blockIdx.x * BN] = threadResults[resIdx];
//        }
    }
#endif
}