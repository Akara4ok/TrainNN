cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TrainNN
        src/main.cpp
        src/Matrix/Matrix.cpp
        src/Matrix/Calculation/CpuMatrixCalculation.cpp src/Matrix/Calculation/GpuMatrixCalculation.cpp
        src/Model/Activation/ReluActivation.cpp src/Model/Activation/SigmoidActivation.cpp
        src/Model/CostFunction/BinaryCrossEntropy.cpp
        src/Model/Layer/Linear.cpp
        src/Model/Model.cpp
        src/Dataset/Dataset.cpp src/Dataset/ImageFlattenDataset.cpp src/Model/Activation/ActivationTypes.cpp src/Model/CostFunction/CostTypes.cpp src/Model/Metrics/Accuracy.cpp)

target_link_libraries(TrainNN ${OpenCV_LIBS})
target_include_directories(TrainNN PUBLIC include/)