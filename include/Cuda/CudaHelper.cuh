//
// Created by vlad on 4/27/23.
//

#ifndef TRAINNN_CUDAHELPER_CUH
#define TRAINNN_CUDAHELPER_CUH

class CudaHelper {
public:
    static const int THREADS_PER_BLOCK = 1024;

    static const int THREAD_PER_TWO_DIM_BLOCK = 32;

    static void calculateLinearThreadNum(int& threads_x, int& blocks_x, int size);

    static void calculateBlockThreadNum(int& threads_x, int& threads_y,
                                        int& blocks_x, int& blocks_y,
                                        int height, int width);

    static void allocateGpuMemory(float** data, int size);

    static void deleteGpuMemory(float* data);

    static void copyFromCpuToGpu(float* cpuData, float* gpuData, int size);

    static void copyFromGpuToCpu(float* gpuData, float* cpuData, int size);

    static void copyFromGpuToGpu(float* src, float* dest, int size);
};


#endif //TRAINNN_CUDAHELPER_CUH
