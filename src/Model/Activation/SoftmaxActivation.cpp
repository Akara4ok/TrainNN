//
// Created by vlad on 4/26/23.
//

#include "Model/Activation/SoftmaxActivation.h"

Matrix::Ptr SoftmaxActivation::calculate(Matrix& matrix) {
    Matrix::Ptr exp_matrix = Matrix::exp(matrix);
    Matrix::Ptr sum_matrix = Matrix::sum(*exp_matrix, 1);
    return *exp_matrix / *sum_matrix;
}

Matrix::Ptr SoftmaxActivation::derivative(Matrix& X, Matrix& dA) {
    Matrix::Ptr softmax = calculate(X);
    Matrix::Ptr sumAdA = Matrix::sum(*(dA * *softmax), 1);
    return *softmax * *(dA - *sumAdA);
}
