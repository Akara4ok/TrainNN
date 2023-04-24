//
// Created by vlad on 4/24/23.
//

#include <limits>
#include "Model/Activation/ReluActivation.h"

Matrix::Ptr ReluActivation::calculate(Matrix& matrix) {
    return Matrix::clip(matrix,
                        0,std::numeric_limits<float>::max(),
                        0, std::numeric_limits<float>::max());
}

Matrix::Ptr ReluActivation::derivative(Matrix &matrix) {
    return Matrix::clip(matrix,
                        0, 0,
                        0, 1);
}
