//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_MATRIX_H
#define CMAKE_AND_CUDA_MATRIX_H

#include <map>
#include <memory>
#include <vector>
#include "config.hpp"

class IMatrixCalculation;

class Matrix {
    float* data = nullptr;
    float* gpuData = nullptr;
    int height = 0;
    int width = 0;
    static Provider lastProvider;
    static std::map<Provider, std::shared_ptr<IMatrixCalculation>> calculation;
    static std::shared_ptr<IMatrixCalculation> currentAlgo;
    bool isUseCpu = false;
    bool isUseGpu = false;
public:
    typedef std::shared_ptr<Matrix> Ptr;

    static void initCalculation();

    Matrix() = default;

    Matrix(int height, int width, Provider initProvider = Provider::CPU);

    Matrix(float* newData, int height, int width, Provider initProvider = Provider::CPU);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(Matrix&& other) noexcept;

    ~Matrix();

    [[nodiscard]] float& get(int rowIndex, int colIndex) const {
        return (data + rowIndex * width)[colIndex];
    };

    [[nodiscard]] int getWidth() const {
        return width;
    };

    [[nodiscard]] int getHeight() const {
        return height;
    };

    [[nodiscard]] float* getData() const {
        return data;
    };

    [[nodiscard]] float* getGpuData() const {
        return gpuData;
    };

    [[nodiscard]] bool getIsUseCpu() const {
        return isUseCpu;
    }

    [[nodiscard]] bool getIsUseGpu() const {
        return isUseGpu;
    }

    void setNewDataWithSize(float* newData, int newHeight, int newWidth);

    void setNewGpuDataWithSize(float* newData, int newHeight, int newWidth);

    void setGpuData(float* newData);

    void copyGpuToCpu();

    void copyCpuToGpu();

    static Matrix copy(const Matrix& other, Provider from, Provider to);

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

    float* operator[](int index) {
        return data + index * width;
    }

    float* operator[](int index) const {
        return data + index * width;
    }

    void reciprocal();

    static Matrix reciprocal(const Matrix& matrix);

    static Matrix::Ptr merge(std::vector<Matrix::Ptr>::iterator begin,
                             std::vector<Matrix::Ptr>::iterator end,
                             int axis);


    static Matrix argmax(const Matrix& matrix, int axis);
};


#endif //CMAKE_AND_CUDA_MATRIX_H