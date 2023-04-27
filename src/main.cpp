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

int main() {
    Config::getInstance().setProvider(Provider::GPU);
//    Provider provider = Config::getInstance().getProvider();
    Matrix matrix(10, 10, Provider::GPU);
    matrix.randomInit(1);
    matrix.copyGpuToCpu();
    std::cout << matrix << "\n";
}