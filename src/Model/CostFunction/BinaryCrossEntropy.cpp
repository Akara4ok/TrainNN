//
// Created by vlad on 4/24/23.
//

#include "Model/CostFunction/BinaryCrossEntropy.h"

float BinaryCrossEntropy::calculate(const Matrix& YHat, const Matrix& Y) {
    Matrix first = Y * Matrix::log(YHat);
    Matrix second = (-Y + 1) * Matrix::log(-YHat + 1);
    return -(first + second).sum() / static_cast<float>(Y.getWidth());
}

Matrix BinaryCrossEntropy::derivative(const Matrix& YHat, const Matrix& Y) {
    Matrix first = -(Y / YHat);
    Matrix second = (-Y + 1) / (-YHat + 1);
    return first + second;
}
