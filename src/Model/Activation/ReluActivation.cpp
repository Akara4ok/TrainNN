//
// Created by vlad on 4/24/23.
//

#include "Model/Activation/ReluActivation.h"

#include <limits>

Matrix ReluActivation::calculate(const Matrix& matrix) {
    return Matrix::clip(matrix,
                        0, std::numeric_limits<float>::max(),
                        0, std::numeric_limits<float>::max());
}

Matrix ReluActivation::derivative(const Matrix& X, const Matrix& dA) {
    return dA * Matrix::clip(X, 0, 0, 0, 1);
}
