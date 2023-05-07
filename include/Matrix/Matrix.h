//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_MATRIX_H
#define CMAKE_AND_CUDA_MATRIX_H

#include "config.hpp"
#include <map>
#include <memory>
#include <vector>

class IMatrixCalculation;

class Matrix {
    float* data;
    int height;
    int width;
    static std::map<Provider, std::unique_ptr<IMatrixCalculation>> calculation;
public:
    typedef std::shared_ptr<Matrix> Ptr;

    static void initCalculation();

    Matrix();

    Matrix(int height, int width);

    Matrix(const float* data, int height, int width);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(Matrix&& other);

    ~Matrix();

    float& get(int rowIndex, int colIndex);

    [[nodiscard]] float get(int rowIndex, int colIndex) const;

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    void setNewDataWithSize(float* new_data, int new_height, int new_width);

    void randomInit(int w);

    void zeroInit();

    friend std::ostream& operator<<(std::ostream& os, const Matrix& obj);

    [[nodiscard]] float sum() const;

    static Matrix sum(const Matrix& matrix, int axis);

    static Matrix multiply(const Matrix& lhs, const Matrix& rhs);

    void exp();

    static Matrix exp(const Matrix& matrix);

    void log();

    static Matrix log(const Matrix& matrix);

    void transpose();

    static Matrix transpose(const Matrix& matrix);

    void clip(float minBound, float maxBound,
              float minValueToSet, float maxValueToSet);

    static Matrix clip(const Matrix& matrix,
                       float minBound, float maxBound,
                       float minValueToSet, float maxValueToSet);

    Matrix operator+(const Matrix& rhs) const;

    Matrix operator+(float value) const;

    Matrix operator-() const;

    Matrix operator-(const Matrix& rhs) const;

    Matrix operator-(float value) const;

    Matrix operator*(const Matrix& rhs) const;

    Matrix operator*(float value) const;

    Matrix operator/(const Matrix& rhs) const;

    Matrix operator/(float value) const;

    float* operator[](int index);

    float* operator[](int index) const;

    void reciprocal();

    static Matrix reciprocal(const Matrix& matrix);

    static Matrix::Ptr merge(std::vector<Matrix::Ptr>::iterator begin,
                             std::vector<Matrix::Ptr>::iterator end,
                             int axis);


    static Matrix argmax(const Matrix& matrix, int axis);
};


#endif //CMAKE_AND_CUDA_MATRIX_H