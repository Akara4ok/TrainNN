//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_CPUMATRIXCALCULATION_H
#define CMAKE_AND_CUDA_CPUMATRIXCALCULATION_H

#include <memory>
#include "IMatrixCalculation.h"

class CpuMatrixCalculation : public IMatrixCalculation {
public:
    typedef std::unique_ptr<CpuMatrixCalculation> Ptr;

    Matrix sum(const Matrix& matrix, int axis) override;

    Matrix multiply(const Matrix& lhs, const Matrix& rhs) override;

    Matrix exp(const Matrix& matrix) override;

    void expInline(Matrix& matrix) override;

    Matrix log(const Matrix& matrix) override;

    void logInline(Matrix& matrix) override;

    Matrix transpose(const Matrix& matrix) override;

    void transposeInline(Matrix& matrix) override;

    Matrix elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) override;

    Matrix elementWiseDivide(const Matrix& lhs, const Matrix& rhs) override;

    Matrix
    clip(const Matrix& matrix,
         float minBound, float maxBound,
         float minValueToSet, float maxValueToSet) override;

    void clipInline(Matrix& matrix,
                    float minBound, float maxBound,
                    float minValueToSet, float maxValueToSet) override;

    Matrix sum(const Matrix& lhs, const Matrix& rhs) override;

    Matrix subtract(const Matrix& lhs, const Matrix& rhs) override;

    Matrix reciprocal(const Matrix& matrix) override;

    void reciprocalInline(Matrix& matrix) override;

    Matrix argmax(const Matrix& matrix, int axis) override;

    void randomInit(Matrix& matrix, int w) override;

    void zeroInit(Matrix& matrix) override;
};


#endif //CMAKE_AND_CUDA_CPUMATRIXCALCULATION_H
