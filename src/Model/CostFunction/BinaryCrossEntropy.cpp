//
// Created by vlad on 4/24/23.
//

#include "Model/CostFunction/BinaryCrossEntropy.h"

float BinaryCrossEntropy::calculate(Matrix &YHat, Matrix &Y) {
    Matrix::Ptr first = Y * *Matrix::log(YHat);
    Matrix::Ptr second = *(*-Y + 1) * *Matrix::log(*(*-YHat + 1));
    return -(*first + *second)->sum() / Y.getWidth();
}

Matrix::Ptr BinaryCrossEntropy::derivative(Matrix &YHat, Matrix &Y) {
    Matrix::Ptr first = - *(Y / YHat);
    Matrix::Ptr second = *(*-Y + 1) / *Matrix::log(*(*-YHat + 1));
    return *first + *second;
}
