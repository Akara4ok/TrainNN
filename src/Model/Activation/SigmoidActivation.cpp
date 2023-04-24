//
// Created by vlad on 4/24/23.
//

#include "Model/Activation/SigmoidActivation.h"

Matrix::Ptr SigmoidActivation::calculate(Matrix& matrix) {
    Matrix::Ptr result = *Matrix::exp(*(-matrix)) + 1;
    result->reciprocal();
    return result;
}

Matrix::Ptr SigmoidActivation::derivative(Matrix &matrix) {
    Matrix::Ptr sigmoid = calculate(matrix);
    Matrix::Ptr negativeSigmoid = -(*sigmoid);
    return Matrix::elementWiseMultiply(*sigmoid, *(*negativeSigmoid + 1));
}