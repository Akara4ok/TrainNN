cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(TrainNN
        src/main.cpp
        src/Matrix/Matrix.cpp
        src/Matrix/Calculation/CpuMatrixCalculation.cpp src/Matrix/Calculation/GpuMatrixCalculation.cpp
        src/Model/Activation/ReluActivation.cpp src/Model/Activation/SigmoidActivation.cpp src/Model/CostFunction/BinaryCrossEntropy.cpp)

target_include_directories(TrainNN PUBLIC include/)