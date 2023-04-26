//
// Created by vlad on 4/24/23.
//

#include "Model/Activation/SigmoidActivation.h"

Matrix SigmoidActivation::calculate(const Matrix& matrix) {
    Matrix result = Matrix::exp(-matrix) + 1;
    result.reciprocal();
    return result;
}

Matrix SigmoidActivation::derivative(const Matrix& X, const Matrix& dA) {
    Matrix sigmoid = calculate(X);
    Matrix negativeSigmoid = -sigmoid;
    Matrix dSigmoid = sigmoid * (negativeSigmoid + 1);
    return dA * dSigmoid;
}