//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include "config.hpp"
#include "Matrix/Matrix.h"

int main(){
    Config::getInstance().setProvider(Provider::CPU);
//    Provider provider = Config::getInstance().getProvider();
    Matrix::Ptr matrix(new Matrix(5, 3));

    for (int i = 0; i < matrix->getHeight(); ++i) {
        for (int j = 0; j < matrix->getWidth(); ++j) {
            matrix->get(i, j) = i * matrix->getWidth() + j;
        }
    }

    std::cout << (*matrix);

    Matrix::Ptr new_matrix(new Matrix(*matrix));
    std::cout << *new_matrix;
    auto product = Matrix::elementWiseMultiply(*matrix, *new_matrix);
    std::cout << *product;
//    matrix->transpose();
//    std::cout << (*matrix);
}