//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_IMATRIXCALCULATION_H
#define CMAKE_AND_CUDA_IMATRIXCALCULATION_H

class Matrix;

class IMatrixCalculation {
public:
    typedef std::unique_ptr<IMatrixCalculation> Ptr;

    virtual std::unique_ptr<Matrix> sum(Matrix& matrix, int axis) = 0;
    virtual std::unique_ptr<Matrix> multiply(Matrix& lhs, Matrix& rhs) = 0;
    virtual std::unique_ptr<Matrix> exp(Matrix& matrix) = 0;
    virtual void exp_inline(Matrix& matrix) = 0;
    virtual std::unique_ptr<Matrix> log(Matrix& matrix) = 0;
    virtual void log_inline(Matrix& matrix) = 0;
    virtual std::unique_ptr<Matrix> transpose(Matrix& matrix) = 0;
    virtual void transpose_inline(Matrix& matrix) = 0;
    virtual std::unique_ptr<Matrix> elementWiseMultiply(Matrix& lhs, Matrix& rhs) = 0;
    virtual std::unique_ptr<Matrix> elementWiseDivide(Matrix& lhs, Matrix& rhs) = 0;
    virtual std::unique_ptr<Matrix> clip(Matrix& matrix,
                                         float minBound, float maxBound,
                                         float minValueToSet, float maxValueToSet) = 0;
    virtual void clip_inline(Matrix &matrix,
                             float minBound, float maxBound,
                             float minValueToSet, float maxValueToSet) = 0;
    virtual std::unique_ptr<Matrix> sum(Matrix& lhs, Matrix& rhs) = 0;
    virtual std::unique_ptr<Matrix> subtract(Matrix& lhs, Matrix& rhs) = 0;
    virtual std::unique_ptr<Matrix> reciprocal(Matrix& matrix) = 0;
    virtual void reciprocal_inline(Matrix& matrix) = 0;
    virtual ~IMatrixCalculation() = default;
};


#endif //CMAKE_AND_CUDA_IMATRIXCALCULATION_H
