//
// Created by vlad on 4/26/23.
//

#include "Model/CostFunction/CrossEntropy.h"

float CrossEntropy::calculate(Matrix& YHat, Matrix& Y) {
    return -(Y * *Matrix::log(YHat))->sum() / Y.getWidth();
}

Matrix::Ptr CrossEntropy::derivative(Matrix& YHat, Matrix& Y) {
    return - *(Y / YHat);
}
