//
// Created by vlad on 4/27/23.
//

#ifndef TRAINNN_CUDAKERNELFUNCTIONS_CUH
#define TRAINNN_CUDAKERNELFUNCTIONS_CUH

namespace GPU{
    __global__ void zeroInit(float* data, int height, int width);
    __global__ void scale(float* data, int height, int width, float value);
}

#endif //TRAINNN_CUDAKERNELFUNCTIONS_CUH
