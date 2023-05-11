//
// Created by vlad on 4/27/23.
//

#ifndef TRAINNN_CUDAFUNCTIONS_CUH
#define TRAINNN_CUDAFUNCTIONS_CUH

namespace GPU {
    const int BLOCK_DIM = 32;

    const int BHL = 64;
    const int BWR = 64;
    const int BWL = 8;
    const int THL = 8;
    const int TWR = 8;

    __global__ void zeroInit(float* data, int height, int width);

    __global__ void multiply(float* data, int height, int width, float value);

    __global__ void sum(float* result, const float* data, int height, int width, int axis);

    __global__ void exp(float* result, const float* data, int height, int width);

    __global__ void log(float* result, const float* data, int height, int width);

    __global__ void argmax(float* result, const float* data, int height, int width, int axis);

    __global__ void reciprocal(float* result, const float* data, int height, int width);

    __global__ void clip(float* result, const float* data, int height, int width,
                         float minBound, float maxBound, float minValueToSet, float maxValueToSet);

    __global__ void transpose(float* result, const float* data, int height, int width);

    __global__ void sum(float* result, const float* lhsData, const float* rhsData,
                        int heightLhs, int widthLhs, int heightRhs, int widthRhs);

    __global__ void subtract(float* result, const float* lhsData, const float* rhsData,
                             int heightLhs, int widthLhs, int heightRhs, int widthRhs);

    __global__ void elementWiseMultiply(float* result, const float* lhsData, const float* rhsData,
                                        int heightLhs, int widthLhs, int heightRhs, int widthRhs);

    __global__ void elementWiseDivide(float* result, const float* lhsData, const float* rhsData,
                                      int heightLhs, int widthLhs, int heightRhs, int widthRhs);

    __global__ void multiply(float* result, const float* lhsData, const float* rhsData,
                             int heightLhs, int widthLhs, int widthRhs);
}

#endif //TRAINNN_CUDAFUNCTIONS_CUH
