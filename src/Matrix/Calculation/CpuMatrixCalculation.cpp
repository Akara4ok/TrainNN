//
// Created by vlad on 4/23/23.
//

#include <cmath>
#include <random>
#include "Matrix/Matrix.h"
#include "Matrix/Calculation/CpuMatrixCalculation.h"

Matrix CpuMatrixCalculation::sum(const Matrix& matrix, int axis) {
    if (axis == -1) {
        Matrix result(1, 1);
        float sum = 0;
        for (int i = 0; i < matrix.getHeight(); ++i) {
            for (int j = 0; j < matrix.getWidth(); ++j) {
                sum += matrix[i][j];
            }
        }
        result[0][0] = sum;
        return result;
    }
    if (axis == 0) {
        Matrix result(matrix.getHeight(), 1);
        for (int i = 0; i < matrix.getHeight(); ++i) {
            float sum = 0;
            for (int j = 0; j < matrix.getWidth(); ++j) {
                sum += matrix[i][j];
            }
            result[i][0] = sum;
        }
        return result;
    }
    if (axis == 1) {
        Matrix result(1, matrix.getWidth());
        for (int i = 0; i < matrix.getWidth(); ++i) {
            float sum = 0;
            for (int j = 0; j < matrix.getHeight(); ++j) {
                sum += matrix[j][i];
            }
            result[0][i] = sum;
        }
        return result;
    }
    return {};
}

Matrix CpuMatrixCalculation::multiply(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.getHeight(), rhs.getWidth());
    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < rhs.getWidth(); ++j) {
            float sum = 0;
            for (int k = 0; k < lhs.getWidth(); ++k) {
                sum += lhs[i][k] * rhs[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

Matrix CpuMatrixCalculation::exp(const Matrix& matrix) {
    Matrix result(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result[i][j] = expf(matrix[i][j]);
        }
    }
    return result;
}

void CpuMatrixCalculation::exp_inline(Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix[i][j] = expf(matrix[i][j]);
        }
    }
}

Matrix CpuMatrixCalculation::log(const Matrix& matrix) {
    Matrix result(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result[i][j] = logf(matrix[i][j]);
        }
    }
    return result;
}

void CpuMatrixCalculation::log_inline(Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix[i][j] = logf(matrix[i][j]);
        }
    }
}

Matrix CpuMatrixCalculation::transpose(const Matrix& matrix) {
    Matrix result(matrix.getWidth(), matrix.getHeight());
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

void CpuMatrixCalculation::transpose_inline(Matrix& matrix) {
    auto* new_data = new float[matrix.getHeight() * matrix.getWidth()];
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            (new_data + j * matrix.getHeight())[i] = matrix[i][j];
        }
    }
    matrix.setNewDataWithSize(new_data, matrix.getWidth(), matrix.getHeight());
}

Matrix CpuMatrixCalculation::elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.getHeight(), lhs.getWidth());

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result[i][j] =
                    lhs[i][j] *
                    rhs[rhs.getHeight() == lhs.getHeight() ? i : 0]
                    [rhs.getWidth() == lhs.getWidth() ? j : 0];
        }
    }
    return result;
}

Matrix CpuMatrixCalculation::elementWiseDivide(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.getHeight(), lhs.getWidth());

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result[i][j] =
                    lhs[i][j] /
                    rhs[rhs.getHeight() == lhs.getHeight() ? i : 0]
                    [rhs.getWidth() == lhs.getWidth() ? j : 0];
        }
    }
    return result;
}

Matrix
CpuMatrixCalculation::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                           float maxValueToSet) {
    Matrix result(matrix);

    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            if (result[i][j] < minBound) {
                result[i][j] = minValueToSet;
            } else if (result[i][j] > maxBound) {
                result[i][j] = maxValueToSet;
            }
        }
    }
    return result;
}

void CpuMatrixCalculation::clip_inline(Matrix& matrix, float minBound, float maxBound, float minValueToSet,
                                       float maxValueToSet) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            if (matrix[i][j] < minBound) {
                matrix[i][j] = minValueToSet;
            } else if (matrix[i][j] >= maxBound) {
                matrix[i][j] = maxValueToSet;
            }
        }
    }
}

Matrix CpuMatrixCalculation::sum(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.getHeight(), lhs.getWidth());

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result[i][j] =
                    lhs[i][j] +
                    rhs[rhs.getHeight() == lhs.getHeight() ? i : 0]
                    [rhs.getWidth() == lhs.getWidth() ? j : 0];
        }
    }

    return result;
}

Matrix CpuMatrixCalculation::subtract(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.getHeight(), lhs.getWidth());

    for (int i = 0; i < lhs.getHeight(); ++i) {
        for (int j = 0; j < lhs.getWidth(); ++j) {
            result[i][j] =
                    lhs[i][j] -
                    rhs[rhs.getHeight() == lhs.getHeight() ? i : 0]
                    [rhs.getWidth() == lhs.getWidth() ? j : 0];
        }
    }

    return result;
}

Matrix CpuMatrixCalculation::reciprocal(const Matrix& matrix) {
    Matrix result(matrix.getHeight(), matrix.getWidth());

    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            result[i][j] = 1 / matrix[i][j];
        }
    }

    return result;
}

void CpuMatrixCalculation::reciprocal_inline(Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix[i][j] = 1 / matrix[i][j];
        }
    }
}

Matrix CpuMatrixCalculation::argmax(const Matrix& matrix, int axis) {
    if (axis == 0) {
        Matrix result(matrix.getHeight(), 1);
        for (int i = 0; i < matrix.getHeight(); ++i) {
            int maxInd = 0;
            float maxValue = -1;
            for (int j = 0; j < matrix.getWidth(); ++j) {
                if (matrix[i][j] > maxValue) {
                    maxValue = matrix[i][j];
                    maxInd = j;
                }
            }
            result[i][0] = static_cast<float>(maxInd);
        }
        return result;
    } else if (axis == 1) {
        Matrix result(1, matrix.getWidth());
        for (int i = 0; i < matrix.getWidth(); ++i) {
            int maxInd = 0;
            float maxValue = -1;
            for (int j = 0; j < matrix.getHeight(); ++j) {
                if (matrix[j][i] > maxValue) {
                    maxValue = matrix[j][i];
                    maxInd = j;
                }
            }
            result[0][i] = static_cast<float>(maxInd);
        }
        return result;
    }
    return {};
}

void CpuMatrixCalculation::randomInit(Matrix& matrix, int w) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> dist{0, 1};

    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix[i][j] = static_cast<float>(dist(gen) * sqrt(2.0 / w));
        }
    }
}

void CpuMatrixCalculation::zeroInit(Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            matrix[i][j] = 0;
        }
    }
}
