//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_IMATRIXCALCULATION_H
#define CMAKE_AND_CUDA_IMATRIXCALCULATION_H

class Matrix;

class IMatrixCalculation {
public:
    typedef std::unique_ptr<IMatrixCalculation> Ptr;

    virtual Matrix sum(const Matrix& matrix, int axis) = 0;

    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) = 0;

    virtual Matrix exp(const Matrix& matrix) = 0;

    virtual void expInline(Matrix& matrix) = 0;

    virtual Matrix log(const Matrix& matrix) = 0;

    virtual void logInline(Matrix& matrix) = 0;

    virtual Matrix transpose(const Matrix& matrix) = 0;

    virtual void transposeInline(Matrix& matrix) = 0;

    virtual Matrix elementWiseMultiply(const Matrix& lhs, const Matrix& rhs) = 0;

    virtual Matrix elementWiseDivide(const Matrix& lhs, const Matrix& rhs) = 0;

    virtual Matrix clip(const Matrix& matrix,
                        float minBound, float maxBound,
                        float minValueToSet, float maxValueToSet) = 0;

    virtual void clipInline(Matrix& matrix,
                            float minBound, float maxBound,
                            float minValueToSet, float maxValueToSet) = 0;

    virtual Matrix sum(const Matrix& lhs, const Matrix& rhs) = 0;

    virtual Matrix subtract(const Matrix& lhs, const Matrix& rhs) = 0;

    virtual Matrix reciprocal(const Matrix& matrix) = 0;

    virtual void reciprocalInline(Matrix& matrix) = 0;

    virtual Matrix argmax(const Matrix& matrix, int axis) = 0;

    virtual void randomInit(Matrix& matrix, int w) = 0;

    virtual void zeroInit(Matrix& matrix) = 0;

    virtual ~IMatrixCalculation() = default;
};


#endif //CMAKE_AND_CUDA_IMATRIXCALCULATION_H
