//
// Created by vlad on 4/26/23.
//

#include "Model/CostFunction/CrossEntropy.h"

float CrossEntropy::calculate(const Matrix& YHat, const Matrix& Y) {
    return -(Y * Matrix::log(YHat)).sum() / static_cast<float>(Y.getWidth());
}

Matrix CrossEntropy::derivative(const Matrix& YHat, const Matrix& Y) {
    return -(Y / YHat);
}
