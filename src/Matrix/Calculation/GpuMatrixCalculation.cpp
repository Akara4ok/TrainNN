//
// Created by vlad on 4/23/23.
//

#include "Matrix/Matrix.h"
#include "Matrix/Calculation/GpuMatrixCalculation.h"

Matrix GpuMatrixCalculation::sum(const Matrix& matrix, int axis) {
    return {};
}

Matrix GpuMatrixCalculation::multiply(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::exp(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::exp_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::log(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::log_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::transpose(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::transpose_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::elementWiseDivide(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix
GpuMatrixCalculation::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                           float maxValueToSet) {
    return {};
}

void GpuMatrixCalculation::clip_inline(Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {

}

Matrix GpuMatrixCalculation::sum(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::subtract(const Matrix& lhs, const Matrix& rhs) {
    return {};
}

Matrix GpuMatrixCalculation::reciprocal(const Matrix& matrix) {
    return {};
}

void GpuMatrixCalculation::reciprocal_inline(Matrix& matrix) {

}

Matrix GpuMatrixCalculation::argmax(const Matrix& matrix, int axis) {
    return {};
}
