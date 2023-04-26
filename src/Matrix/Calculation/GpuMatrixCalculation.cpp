//
// Created by vlad on 4/23/23.
//

#include "Matrix/Matrix.h"
#include "Matrix/Calculation/GpuMatrixCalculation.h"

Matrix::Ptr GpuMatrixCalculation::sum(Matrix& matrix, int axis) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::multiply(Matrix& lhs, Matrix& rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::exp(Matrix& matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::exp_inline(Matrix& matrix) {

}

Matrix::Ptr GpuMatrixCalculation::log(Matrix& matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::log_inline(Matrix& matrix) {

}

Matrix::Ptr GpuMatrixCalculation::transpose(Matrix& matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::transpose_inline(Matrix& matrix) {

}

Matrix::Ptr GpuMatrixCalculation::elementWiseMultiply(Matrix& lhs, Matrix& rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::elementWiseDivide(Matrix& lhs, Matrix& rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr
GpuMatrixCalculation::clip(Matrix& matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::clip_inline(Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {

}

Matrix::Ptr GpuMatrixCalculation::sum(Matrix& lhs, Matrix& rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::subtract(Matrix& lhs, Matrix& rhs) {
    return Matrix::Ptr();
}

Matrix::Ptr GpuMatrixCalculation::reciprocal(Matrix& matrix) {
    return Matrix::Ptr();
}

void GpuMatrixCalculation::reciprocal_inline(Matrix& matrix) {

}

std::shared_ptr<Matrix> GpuMatrixCalculation::argmax(Matrix& matrix, int axis) {
    return std::shared_ptr<Matrix>();
}
