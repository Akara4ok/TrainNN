//
// Created by vlad on 4/23/23.
//

#include <stdio.h>
#include "CUDAProvider/test_helpers.cuh"

__global__ void gpu_hello_cuda(){
    printf("Hello from GPU\n");
}
