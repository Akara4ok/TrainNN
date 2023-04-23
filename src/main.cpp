//
// Created by vlad on 4/23/23.
//

#include <stdio.h>
#include "CUDAProvider/test.cuh"
int main(){
    printf("Hello from CPU\n");
    gpu_hello();
}