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

std::unique_ptr<Matrix>
GpuMatrixCalculation::clip(Matrix &matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    return std::unique_ptr<Matrix>();
}

void GpuMatrixCalculation::clip_inline(Matrix &matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {

}

std::unique_ptr<Matrix> GpuMatrixCalculation::sum(Matrix &lhs, Matrix &rhs) {
    return std::unique_ptr<Matrix>();
}

std::unique_ptr<Matrix> GpuMatrixCalculation::subtract(Matrix &lhs, Matrix &rhs) {
    return std::unique_ptr<Matrix>();
}

std::unique_ptr<Matrix> GpuMatrixCalculation::reciprocal(Matrix &matrix) {
    return std::unique_ptr<Matrix>();
}

void GpuMatrixCalculation::reciprocal_inline(Matrix &matrix) {

}
