//
// Created by vlad on 4/26/23.
//

#include "Model/Activation/SoftmaxActivation.h"

Matrix SoftmaxActivation::calculate(const Matrix& matrix) {
    Matrix exp_matrix = Matrix::exp(matrix);
    Matrix sum_matrix = Matrix::sum(exp_matrix, 1);
    Matrix divide_matrix = exp_matrix / sum_matrix;
    return exp_matrix / sum_matrix;
}

Matrix SoftmaxActivation::derivative(const Matrix& X, const Matrix& dA) {
    Matrix softmax = calculate(X);
    Matrix sumAdA = Matrix::sum((dA * softmax), 1);
    return softmax * (dA - sumAdA);
}
