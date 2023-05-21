//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaFunctions.cuh"

namespace GPU {
    __global__ void zeroInit(float* data, int height, int width) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            data[row * width + col] = 0;
        }
    }

    __global__ void multiply(float* data, int height, int width, float value) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            data[row * width + col] = data[row * width + col] * value;
        }
    }

#ifdef CUDA_STANDARD_SUM

    __global__ void sum(float* result, const float* data, int height, int width, int axis) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
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

#endif
#ifdef CUDA_SHARED_SUM

    __global__ void sum(float* result, const float* data, int height, int width, int axis) {
        __shared__ float dataTile[BLOCK_DIM][BLOCK_DIM];

        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

        dataTile[threadIdx.y][threadIdx.x] = 0;
        if ((row < height) && (col < width)) {
            dataTile[threadIdx.y][threadIdx.x] = data[row * width + col];
        }

        __syncthreads();

        if ((row < height) && (col < width)) {
            float subSum = 0;
            if (axis == -1 && threadIdx.x == 0) {
                for (int i = 0; i < blockDim.x; i++) {
                    subSum += dataTile[threadIdx.y][i];
                }
                atomicAdd(result, subSum);
            }
            if (axis == 0 && threadIdx.x == 0) {
                for (int i = 0; i < blockDim.x; i++) {
                    subSum += dataTile[threadIdx.y][i];
                }
                atomicAdd(result + row, subSum);
            }
            if (axis == 1 && threadIdx.y == 0) {
                for (int i = 0; i < blockDim.y; i++) {
                    subSum += dataTile[i][threadIdx.x];
                }
                atomicAdd(result + col, subSum);
            }
        }
    }

#endif

    __global__ void exp(float* result, const float* data, int height, int width) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = expf(data[row * width + col]);
        }
    }

    __global__ void log(float* result, const float* data, int height, int width) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = logf(data[row * width + col]);
        }
    }

    __global__ void argmax(float* result, const float* data, int height, int width, int axis) {
        const unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
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
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[row * width + col] = 1 / data[row * width + col];
        }
    }

    __global__ void clip(float* result, const float* data, int height, int width,
                         float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
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

#ifdef CUDA_STANDARD_TRANSPOSE

    __global__ void transpose(float* result, const float* data, int height, int width) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < height) && (col < width)) {
            result[col * height + row] = data[row * width + col];
        }
    }

#endif
#ifdef CUDA_SHARED_TRANSPOSE

    __global__ void transpose(float* result, const float* data, int height, int width) {
        __shared__ float dataTile[BLOCK_DIM][BLOCK_DIM];

        uint col = blockIdx.x * BLOCK_DIM + threadIdx.x;
        uint row = blockIdx.y * BLOCK_DIM + threadIdx.y;

        int totalTiles = BLOCK_DIM / BWL;

        for (int j = 0; j < totalTiles; j++) {
            int currentCol = j * BWL;
            if ((row + currentCol) < height && col < width) {
                dataTile[threadIdx.y + currentCol][threadIdx.x] = data[(row + currentCol) * width + col];
            }
        }

        __syncthreads();

        const uint trow = blockIdx.x * BLOCK_DIM + threadIdx.y;
        const uint tcol = blockIdx.y * BLOCK_DIM + threadIdx.x;

        for (int j = 0; j < totalTiles; j++) {
            int currentCol = j * BWL;
            if (tcol < height && (trow + currentCol) < width) {
                result[(trow + currentCol) * height + tcol] = dataTile[threadIdx.x][threadIdx.y + currentCol];
            }
        }
    }

#endif
#ifdef CUDA_NO_BANK_TRANSPOSE

    __global__ void transpose(float* result, const float* data, int height, int width) {
        __shared__ float dataTile[BLOCK_DIM][BLOCK_DIM + 1];

        uint col = blockIdx.x * BLOCK_DIM + threadIdx.x;
        uint row = blockIdx.y * BLOCK_DIM + threadIdx.y;

        int totalTiles = BLOCK_DIM / BWL;

        for (int j = 0; j < totalTiles; j++) {
            int currentCol = j * BWL;
            if ((row + currentCol) < height && col < width) {
                dataTile[threadIdx.y + currentCol][threadIdx.x] = data[(row + currentCol) * width + col];
            }
        }

        __syncthreads();

        const uint trow = blockIdx.x * BLOCK_DIM + threadIdx.y;
        const uint tcol = blockIdx.y * BLOCK_DIM + threadIdx.x;

        for (int j = 0; j < totalTiles; j++) {
            int currentCol = j * BWL;
            if (tcol < height && (trow + currentCol) < width) {
                result[(trow + currentCol) * height + tcol] = dataTile[threadIdx.x][threadIdx.y + currentCol];
            }
        }
    }

#endif

    __global__ void
    sum(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs, int heightRhs,
        int widthRhs) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] + rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    subtract(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs, int heightRhs,
             int widthRhs) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] - rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    elementWiseMultiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
                        int heightRhs, int widthRhs) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < heightLhs) && (col < widthLhs)) {
            const unsigned int rowRhs = heightRhs == heightLhs ? row : 0;
            const unsigned int colRhs = widthRhs == widthLhs ? col : 0;
            result[row * widthLhs + col] = lhsData[row * widthLhs + col] * rhsData[rowRhs * widthRhs + colRhs];
        }
    }

    __global__ void
    elementWiseDivide(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
                      int heightRhs, int widthRhs) {
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
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
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if ((row < heightLhs) && (col < widthRhs)) {
            float acc = 0.0;
            for (int i = 0; i < widthLhs; i++) {
                acc += lhsData[row * widthLhs + i] * rhsData[i * widthRhs + col];
            }
            result[row * widthRhs + col] = acc;
        }
    }

#endif
#ifdef CUDA_COALESCING_MULT

    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const uint row = blockIdx.y * BLOCK_DIM + (threadIdx.x / BLOCK_DIM);
        const uint col = blockIdx.x * BLOCK_DIM + (threadIdx.x % BLOCK_DIM);
        if ((row < heightLhs) && (col < widthRhs)) {
            float acc = 0.0;
            for (int i = 0; i < widthLhs; i++) {
                acc += lhsData[row * widthLhs + i] * rhsData[i * widthRhs + col];
            }
            result[row * widthRhs + col] = acc;
        }
    }

#endif
#ifdef CUDA_SHARED_BLOCK_MULT

    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const uint tRow = threadIdx.x / BLOCK_DIM;
        const uint tCol = threadIdx.x % BLOCK_DIM;

        const uint row = blockIdx.y * BLOCK_DIM + tRow;
        const uint col = blockIdx.x * BLOCK_DIM + tCol;

        __shared__ float aTile[BLOCK_DIM][BLOCK_DIM];
        __shared__ float bTile[BLOCK_DIM][BLOCK_DIM];

        float sum = 0;
        const int tiles = (BLOCK_DIM + widthLhs - 1) / BLOCK_DIM;

        uint colLhsOffset = tCol;
        uint rowRhsOffset = tRow;
        for (int tile = 0; tile < tiles; tile++) {
            aTile[tRow][tCol] = 0;
            bTile[tRow][tCol] = 0;
            if (colLhsOffset < widthLhs && row < heightLhs)
                aTile[tRow][tCol] = lhsData[row * widthLhs + colLhsOffset];
            if (rowRhsOffset < widthLhs && col < widthRhs)
                bTile[tRow][tCol] = rhsData[rowRhsOffset * widthRhs + col];

            __syncthreads();

            for (int i = 0; i < BLOCK_DIM; i++) {
                sum += aTile[tRow][i] * bTile[i][tCol];
            }

            __syncthreads();

            colLhsOffset += BLOCK_DIM;
            rowRhsOffset += BLOCK_DIM;
        }
        if ((row < heightLhs) && (col < widthRhs)) {
            result[row * widthRhs + col] = sum;
        }
    }

#endif
#ifdef CUDA_SHARED1D_MULT

    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {
        const uint tRow = threadIdx.x / BWR;
        const uint tCol = threadIdx.x % BWR;

        __shared__ float aTile[BHL * BWL];
        __shared__ float bTile[BWL * BWR];

        const uint innerColLhs = threadIdx.x % BWL;
        const uint innerRowLhs = threadIdx.x / BWL;
        const uint innerColRhs = threadIdx.x % BWR;
        const uint innerRowRhs = threadIdx.x / BWR;

        float localRes[THL]{};

        uint rowLhs = blockIdx.y * BHL + innerRowLhs;
        lhsData += rowLhs * widthLhs + innerColLhs;

        uint colRhs = innerColRhs + blockIdx.x * BWR;
        rhsData += innerRowRhs * widthRhs + colRhs;

        uint rowResOffset = tRow * THL + blockIdx.y * BHL;
        uint resCol = tCol + blockIdx.x * BWR;
        result += rowResOffset * widthRhs + resCol;

        uint aTileIdx = innerRowLhs * BWL + innerColLhs;
        uint bTileIdx = innerRowRhs * BWR + innerColRhs;

        for (uint blockId = 0; blockId < widthLhs; blockId += BWL) {
            aTile[aTileIdx] = 0;
            bTile[bTileIdx] = 0;
            if (rowLhs < heightLhs && blockId + innerColLhs < widthLhs) {
                aTile[aTileIdx] = lhsData[0];
            }
            if (blockId + innerRowRhs < widthLhs && colRhs < widthRhs) {
                bTile[bTileIdx] = rhsData[0];
            }

            __syncthreads();

            for (uint subMatrixIndex = 0; subMatrixIndex < BWL; subMatrixIndex++) {
                float temp = bTile[subMatrixIndex * BWR + tCol];
                for (uint resI = 0; resI < THL; resI++) {
                    localRes[resI] += aTile[(tRow * THL + resI) * BWL + subMatrixIndex] * temp;
                }
            }
            __syncthreads();

            lhsData += BWL;
            rhsData += BWL * widthRhs;
        }

        for (uint resI = 0; resI < THL; resI++) {
            if ((rowResOffset + resI) < heightLhs && resCol < widthRhs) {
                result[0] = localRes[resI];
                result += widthRhs;
            }
        }
    }

#endif
#ifdef CUDA_SHARED2D_MULT

    __global__ void
    multiply(float* result, const float* lhsData, const float* rhsData, int heightLhs, int widthLhs,
             int widthRhs) {

        const uint totalResultsBlockTile = BHL * BWR;
        const uint numThreadsBlockTile = totalResultsBlockTile / (THL * TWR);

        const uint tRow = threadIdx.x / (BWR / TWR);
        const uint tCol = threadIdx.x % (BWR / TWR);

        __shared__ float aTile[BHL * BWL];
        __shared__ float bTile[BWL * BWR];

        const uint innerColLhs = threadIdx.x % BWL;
        const uint innerRowLhs = threadIdx.x / BWL;
        const uint innerColRhs = threadIdx.x % BWR;
        const uint innerRowRhs = threadIdx.x / BWR;

        const uint paddingLhs = numThreadsBlockTile / BWL;
        const uint paddingRhs = numThreadsBlockTile / BWR;

        float localRes[THL * TWR]{};

        float regI[THL]{};
        float regJ[TWR]{};

        uint aTileOffset = innerRowLhs * BWL + innerColLhs;
        uint rowLhsOffset = innerRowLhs + blockIdx.y * BHL;
        lhsData += rowLhsOffset * widthLhs + innerColLhs;

        uint bTileOffset = innerRowRhs * BWR + innerColRhs;
        uint colRhsOffset = innerColRhs + blockIdx.x * BWR;
        rhsData += innerRowRhs * widthRhs + colRhsOffset;

        uint rowResOffset = tRow * THL + blockIdx.y * BHL;
        uint rowColOffset = tCol * TWR + blockIdx.x * BWR;
        result += rowResOffset * widthRhs + rowColOffset;

        for (uint blockId = 0; blockId < widthLhs; blockId += BWL) {
            for (uint innerOffset = 0; innerOffset < BHL; innerOffset += paddingLhs) {
                aTile[aTileOffset + innerOffset * BWL] = 0;
                if (rowLhsOffset + innerOffset < heightLhs && blockId + innerColLhs < widthLhs) {
                    aTile[aTileOffset + innerOffset * BWL] = lhsData[innerOffset * widthLhs + blockId];
                }
            }
            for (uint loadOffset = 0; loadOffset < BWL; loadOffset += paddingRhs) {
                bTile[bTileOffset + loadOffset * BWR] = 0;
                if (blockId + innerRowRhs + loadOffset < widthLhs && colRhsOffset < widthRhs) {
                    bTile[bTileOffset + loadOffset * BWR] = rhsData[(blockId + loadOffset) * widthRhs];
                }
            }

            __syncthreads();

            for (uint subMatrixIndex = 0; subMatrixIndex < BWL; ++subMatrixIndex) {
                for (uint i = 0; i < THL; ++i) {
                    regI[i] = aTile[(tRow * THL + i) * BWL + subMatrixIndex];
                }
                for (uint i = 0; i < TWR; ++i) {
                    regJ[i] = bTile[subMatrixIndex * BWR + tCol * TWR + i];
                }
                for (uint resI = 0; resI < THL; ++resI) {
                    for (uint resJ = 0; resJ < TWR; ++resJ) {
                        localRes[resI * TWR + resJ] += regI[resI] * regJ[resJ];
                    }
                }
            }
            __syncthreads();
        }

        for (uint resI = 0; resI < THL; resI++) {
            for (uint resJ = 0; resJ < TWR; resJ++) {
                if (rowResOffset + resI < heightLhs && rowColOffset + resJ < widthRhs) {
                    result[resI * widthRhs + resJ] = localRes[resI * TWR + resJ];
                }
            }
        }
    }

#endif
}