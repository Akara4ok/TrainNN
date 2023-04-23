//
// Created by vlad on 4/23/23.
//
#include "CUDAProvider/test.cuh"
#include "CUDAProvider/test_helpers.cuh"

void gpu_hello(){
    gpu_hello_cuda<<<1,1>>>();
    cudaDeviceSynchronize();
}