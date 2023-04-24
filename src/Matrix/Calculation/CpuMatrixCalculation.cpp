//
// Created by vlad on 4/23/23.
//

#include <math.h>
#include "Matrix/Matrix.h"
#include "Matrix/Calculation/CpuMatrixCalculation.h"

Matrix::Ptr CpuMatrixCalculation::sum(Matrix &matrix, int axis) {
    if(axis == -1){
        Matrix::Ptr result(new Matrix(1, 1));
        float sum = 0;
        for (int i = 0; i < matrix.getHeight(); ++i) {
            for (int j = 0; j < matrix.getWidth(); ++j) {
                sum += matrix.get(i, j);
            }
        }
        result->get(0, 0) = sum;
        return result;
    }
    if(axis == 0){
        Matrix::Ptr result(new Matrix(matrix.getHeight(), 1));
        for (int i = 0; i < matrix.getHeight(); ++i) {
            float sum = 0;
            for (int j = 0; j < matrix.getWidth(); ++j) {
                sum += matrix.get(i, j);
            }
            result->get(i, 0) = sum;
        }
        return result;
    }
    if(axis == 1){
        Matrix::Ptr result(new Matrix(1, matrix.getWidth()));
        for (int i = 0; i < matrix.getWidth(); ++i) {
            float sum = 0;
            for (int j = 0; j < matrix.getHeight(); ++j) {
                sum += matrix.get(j, i);
            }
            result->get(0, i) = sum;
        }
        return result;
    }
    return Matrix::Ptr();
}

Matrix::Ptr CpuMatrixCalculation::multiply(Matrix &lhs, Matrix &rhs) {
    Matrix::Ptr result(new Matrix(lhs.getHeight(), rhs.getWidth()));
    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < rhs.getWidth(); ++j) {
            float sum = 0;
            for (int k = 0; k < lhs.getWidth(); ++k) {
                sum += lhs.get(i, k) * rhs.get(k, j);
            }
            result->get(i, j) = sum;
        }
    }
    return result;
}

Matrix::Ptr CpuMatrixCalculation::exp(Matrix &matrix) {
    Matrix::Ptr result(new Matrix(matrix.getHeight(), matrix.getWidth()));
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result->get(i, j) = expf(matrix.get(i, j));
        }
    }
    return result;
}

void CpuMatrixCalculation::exp_inline(Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix.get(i, j) = expf(matrix.get(i, j));
        }
    }
}

Matrix::Ptr CpuMatrixCalculation::log(Matrix &matrix) {
    Matrix::Ptr result(new Matrix(matrix.getHeight(), matrix.getWidth()));
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result->get(i, j) = logf(matrix.get(i, j));
        }
    }
    return result;
}

void CpuMatrixCalculation::log_inline(Matrix &matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix.get(i, j) = logf(matrix.get(i, j));
        }
    }
}

Matrix::Ptr CpuMatrixCalculation::transpose(Matrix &matrix) {
    Matrix::Ptr result(new Matrix(matrix.getWidth(), matrix.getHeight()));
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result->get(j, i) = matrix.get(i, j);
        }
    }
    return result;
}

void CpuMatrixCalculation::transpose_inline(Matrix& matrix) {
    float* new_data = new float[matrix.getHeight() * matrix.getWidth()];
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            (new_data + j * matrix.getHeight())[i] = matrix.get(i, j);
        }
    }
    matrix.setNewDataWithSize(new_data, matrix.getWidth(), matrix.getHeight());
}

Matrix::Ptr CpuMatrixCalculation::elementWiseMultiply(Matrix &lhs, Matrix &rhs) {
    Matrix::Ptr result(new Matrix(lhs.getHeight(), lhs.getWidth()));

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result->get(i, j) =
                    lhs.get(i, j) *
                    rhs.get(rhs.getHeight() == lhs.getHeight() ? i : 0,
                            rhs.getWidth() == lhs.getWidth() ? j : 0);
        }
    }
    return result;
}

std::unique_ptr<Matrix> CpuMatrixCalculation::elementWiseDivide(Matrix &lhs, Matrix &rhs) {
    Matrix::Ptr result(new Matrix(lhs.getHeight(), lhs.getWidth()));

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result->get(i, j) =
                    lhs.get(i, j) /
                    rhs.get(rhs.getHeight() == lhs.getHeight() ? i : 0,
                            rhs.getWidth() == lhs.getWidth() ? j : 0);
        }
    }
    return result;
}

std::unique_ptr<Matrix>
CpuMatrixCalculation::clip(Matrix &matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    Matrix::Ptr result(new Matrix(matrix));

    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            if(result->get(i, j) < minBound){
                result->get(i, j) = minValueToSet;
            }
            if(result->get(i, j) > maxBound){
                result->get(i, j) = maxValueToSet;
            }
        }
    }
    return result;
}

void CpuMatrixCalculation::clip_inline(Matrix &matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            if(matrix.get(i, j) < minBound){
                matrix.get(i, j) = minValueToSet;
            }
            if(matrix.get(i, j) >= maxBound){
                matrix.get(i, j) = maxValueToSet;
            }
        }
    }
}

std::unique_ptr<Matrix> CpuMatrixCalculation::sum(Matrix &lhs, Matrix &rhs) {
    Matrix::Ptr result(new Matrix(lhs.getHeight(), lhs.getWidth()));

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result->get(i, j) =
                    lhs.get(i, j) +
                    rhs.get(rhs.getHeight() == lhs.getHeight() ? i : 0,
                            rhs.getWidth() == lhs.getWidth() ? j : 0);
        }
    }

    return result;
}

std::unique_ptr<Matrix> CpuMatrixCalculation::subtract(Matrix &lhs, Matrix &rhs) {
    Matrix::Ptr result(new Matrix(lhs.getHeight(), lhs.getWidth()));

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result->get(i, j) =
                    lhs.get(i, j) -
                    rhs.get(rhs.getHeight() == lhs.getHeight() ? i : 0,
                            rhs.getWidth() == lhs.getWidth() ? j : 0);
        }
    }

    return result;
}

std::unique_ptr<Matrix> CpuMatrixCalculation::reciprocal(Matrix &matrix) {
    Matrix::Ptr result(new Matrix(matrix.getHeight(), matrix.getWidth()));

    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result->get(i, j) = 1 / matrix.get(i, j);
        }
    }

    return result;
}

void CpuMatrixCalculation::reciprocal_inline(Matrix &matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix.get(i, j) = 1 / matrix.get(i, j);
        }
    }
}