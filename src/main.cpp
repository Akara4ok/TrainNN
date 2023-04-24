//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include "config.hpp"
#include "Matrix/Matrix.h"
#include "Model/Activation/ReluActivation.h"
#include "Model/Activation/SigmoidActivation.h"

int main(){
    Config::getInstance().setProvider(Provider::CPU);
//    Provider provider = Config::getInstance().getProvider();
    Matrix::Ptr matrix(new Matrix(5, 3));

    for (int i = 0; i < matrix->getHeight(); ++i) {
        for (int j = 0; j < matrix->getWidth(); ++j) {
            matrix->get(i, j) = i * matrix->getWidth() + j;
            if(i % 2 == 0){
                matrix->get(i, j) = -matrix->get(i, j);
            }
        }
    }

    std::cout << (*matrix);

    IActivation::Ptr sigmoid = SigmoidActivation::Ptr(new SigmoidActivation());
    auto res = sigmoid->calculate(*matrix);
    std::cout << *res;
}