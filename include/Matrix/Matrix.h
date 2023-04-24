//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_MATRIX_H
#define CMAKE_AND_CUDA_MATRIX_H

#include "config.hpp"
#include <map>
#include <memory>

class IMatrixCalculation;

class Matrix {
    float* data;
    int height;
    int width;
    static std::map<Provider, std::unique_ptr<IMatrixCalculation>> calculation;
public:
    typedef std::unique_ptr<Matrix> Ptr;

    static void initCalculation();

    Matrix(int height, int width);
    Matrix(Matrix &other);
    ~Matrix();

    float& get(int index);
    float& get(int rowIndex, int colIndex);
    int getWidth() const;
    int getHeight() const;

    void setNewDataWithSize(float* new_data, int new_height, int new_width);

    friend std::ostream& operator<<(std::ostream& os, Matrix& obj);

    float sum();
    static Matrix::Ptr sum(Matrix& matrix, int axis);

    static Matrix::Ptr multiply(Matrix& lhs, Matrix& rhs);

    void exp();
    static Matrix::Ptr exp(Matrix& matrix);

    void log();
    static Matrix::Ptr log(Matrix& matrix);

    void transpose();
    static Matrix::Ptr transpose(Matrix& matrix);

    void clip(float minBound, float maxBound,
              float minValueToSet, float maxValueToSet);
    static Matrix::Ptr clip(Matrix &matrix,
                    float minBound, float maxBound,
                    float minValueToSet, float maxValueToSet);

    Matrix::Ptr operator+ (Matrix& rhs);
    Matrix::Ptr operator+ (float value);
    Matrix::Ptr operator-();
    Matrix::Ptr operator- (Matrix& rhs);
    Matrix::Ptr operator- (float value);
    Matrix::Ptr operator* (Matrix& rhs);
    Matrix::Ptr operator* (float value);
    Matrix::Ptr operator/ (Matrix& rhs);
    Matrix::Ptr operator/ (float value);
    void reciprocal ();
    static Matrix::Ptr reciprocal (Matrix& matrix);
};


#endif //CMAKE_AND_CUDA_MATRIX_H