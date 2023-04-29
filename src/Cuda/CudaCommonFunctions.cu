//
// Created by vlad on 4/27/23.
//

#include "Cuda/CudaCommonFunctions.cuh"

namespace GPU {
    __global__ void zeroInit(float* data, int height, int width) {
        const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
        const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
        if ((row < height) && (col < width)) {
            data[row * width + col] = 0;
        }
    }
}