cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(TrainNN
        src/main.cpp
        src/CUDAProvider/test.cu src/CUDAProvider/test_helpers.cu)

target_include_directories(TrainNN PUBLIC include)