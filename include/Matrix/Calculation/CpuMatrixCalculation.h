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

    std::unique_ptr<Matrix> sum(Matrix &matrix, int axis) override;
    std::unique_ptr<Matrix> multiply(Matrix &lhs, Matrix &rhs) override;
    std::unique_ptr<Matrix> exp(Matrix &matrix) override;
    void exp_inline(Matrix& matrix) override;
    std::unique_ptr<Matrix> log(Matrix &matrix) override;
    void log_inline(Matrix& matrix) override;
    std::unique_ptr<Matrix> transpose(Matrix &matrix) override;
    void transpose_inline(Matrix& matrix) override;
    std::unique_ptr<Matrix> elementWiseMultiply(Matrix &lhs, Matrix &rhs) override;
    std::unique_ptr<Matrix> elementWiseDivide(Matrix& lhs, Matrix& rhs) override;
    std::unique_ptr<Matrix>
    clip(Matrix &matrix,
         float minBound, float maxBound,
         float minValueToSet, float maxValueToSet) override;
    void clip_inline(Matrix &matrix,
                     float minBound, float maxBound,
                     float minValueToSet, float maxValueToSet) override;
    std::unique_ptr<Matrix> sum(Matrix& lhs, Matrix& rhs) override;
    std::unique_ptr<Matrix> subtract(Matrix& lhs, Matrix& rhs) override;
    std::unique_ptr<Matrix> reciprocal(Matrix &matrix) override;
    void reciprocal_inline(Matrix &matrix) override;
};


#endif //CMAKE_AND_CUDA_CPUMATRIXCALCULATION_H
