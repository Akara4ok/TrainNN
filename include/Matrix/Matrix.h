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
    float* data = nullptr;
    float* gpuData = nullptr;
    int height = 0;
    int width = 0;
    static std::map<Provider, std::shared_ptr<IMatrixCalculation>> calculation;
    std::shared_ptr<IMatrixCalculation> currentAlgo;
    bool isUseCpu = false;
    bool isUseGpu = false;
public:
    typedef std::shared_ptr<Matrix> Ptr;

    static void initCalculation();

    Matrix() = default;

    Matrix(int height, int width, Provider initProvider = Provider::CPU);

    Matrix(float* data, int height, int width, Provider initProvider = Provider::CPU);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(Matrix&& other) noexcept;

    ~Matrix();

    float& get(int rowIndex, int colIndex);

    [[nodiscard]] float get(int rowIndex, int colIndex) const;

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] float* getData() const;

    [[nodiscard]] float* getGpuData() const;

    float* getGpuData();

    void setNewDataWithSize(float* new_data, int new_height, int new_width);

    void setNewGpuDataWithSize(float* new_data, int new_height, int new_width);

    void setGpuData(float* new_data);

    void copyGpuToCpu();

    void copyCpuToGpu();

    void moveCpuToGpu();

    void moveGpuToCpu();

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