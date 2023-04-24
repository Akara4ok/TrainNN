//
// Created by vlad on 4/23/23.
//

#include "Matrix/Matrix.h"
#include "Matrix/Calculation/GpuMatrixCalculation.h"

Matrix::Ptr GpuMatrixCalculation::sum(Matrix &matrix, int axis) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::multiply(Matrix &lhs, Matrix &rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::exp(Matrix &matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::exp_inline(Matrix &matrix) {

}

Matrix::Ptr GpuMatrixCalculation::log(Matrix &matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::log_inline(Matrix &matrix) {

}

Matrix::Ptr GpuMatrixCalculation::transpose(Matrix &matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::transpose_inline(Matrix &matrix) {

}

Matrix::Ptr GpuMatrixCalculation::elementWiseMultiply(Matrix &lhs, Matrix &rhs) {
    return Matrix::Ptr();
}
