//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <memory>
#include <algorithm>
#include "config.hpp"
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Dataset/ImageFlattenDataset.h"
#include "Model/Activation/SoftmaxActivation.h"
#include <limits>

int main() {
    Config::getInstance().setProvider(Provider::GPU);
//    Provider provider = Config::getInstance().getProvider();
    Matrix matrix1(10, 10, Provider::GPU);
    matrix1.zeroInit();
//    Matrix matrix2(1, 10, Provider::GPU);
//    matrix2.zeroInit();
//    Matrix m = Matrix::reciprocal(matrix1);
//    m.copyGpuToCpu();
    matrix1.exp();
    matrix1.copyGpuToCpu();
//    matrix2.copyGpuToCpu();
    std::cout << matrix1 << "\n";
//    std::cout << matrix2 << "\n";
//    std::cout << m << "\n";
}