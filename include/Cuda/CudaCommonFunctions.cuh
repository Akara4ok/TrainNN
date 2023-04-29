//
// Created by vlad on 4/27/23.
//

#ifndef TRAINNN_CUDACOMMONFUNCTIONS_CUH
#define TRAINNN_CUDACOMMONFUNCTIONS_CUH

namespace GPU {
    __global__ void zeroInit(float* data, int height, int width);
}

#endif //TRAINNN_CUDACOMMONFUNCTIONS_CUH
