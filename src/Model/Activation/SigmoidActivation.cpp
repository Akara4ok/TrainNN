//
// Created by vlad on 4/24/23.
//

#include "Model/Activation/SigmoidActivation.h"

Matrix::Ptr SigmoidActivation::calculate(Matrix& matrix) {
    Matrix::Ptr result = *Matrix::exp(*(-matrix)) + 1;
    result->reciprocal();
    return result;
}

Matrix::Ptr SigmoidActivation::derivative(Matrix& X, Matrix& dA) {
    Matrix::Ptr sigmoid = calculate(X);
    Matrix::Ptr negativeSigmoid = -(*sigmoid);
    Matrix::Ptr dSigmoid = (*sigmoid) * (*(*negativeSigmoid + 1));
    return dA * *dSigmoid;
}