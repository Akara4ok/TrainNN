cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TrainNN LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TrainNN
        src/main.cpp
        src/Matrix/Matrix.cpp
        src/Matrix/Calculation/CpuMatrixCalculation.cpp src/Matrix/Calculation/GpuMatrixCalculation.cu
        src/Model/Activation/ActivationTypes.cpp
        src/Model/Activation/ReluActivation.cpp src/Model/Activation/SigmoidActivation.cpp
        src/Model/Activation/SoftmaxActivation.cpp
        src/Model/CostFunction/CostTypes.cpp
        src/Model/CostFunction/BinaryCrossEntropy.cpp src/Model/CostFunction/CrossEntropy.cpp
        src/Model/Layer/Linear.cpp
        src/Model/Model.cpp
        src/Dataset/Dataset.cpp src/Dataset/ImageFlattenDataset.cpp
        src/Model/Metrics/Accuracy.cpp
        src/Cuda/CudaFunctions.cu src/Cuda/CudaHelper.cu
        src/Model/Monitoring/Monitoring.cpp src/Model/Monitoring/LogSample.cpp)


target_link_libraries(TrainNN ${OpenCV_LIBS} -lcurand)
target_include_directories(TrainNN PUBLIC include/)
target_compile_definitions(TrainNN PUBLIC CUDA_SHARED_MULT)
