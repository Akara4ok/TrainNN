//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <random>
#include "Matrix/Matrix.h"
#include "Matrix/Calculation/CpuMatrixCalculation.h"
#include "Matrix/Calculation/GpuMatrixCalculation.h"

std::map<Provider, IMatrixCalculation::Ptr> Matrix::calculation;

void Matrix::initCalculation() {
    calculation.emplace(Provider::CPU, std::make_unique<CpuMatrixCalculation>());
    calculation.emplace(Provider::GPU, std::make_unique<GpuMatrixCalculation>());
}

static int initCalculationCaller = []() {
    Matrix::initCalculation();
    return 0;
}();

Matrix::Matrix() {
    height = 0;
    width = 0;
    data = nullptr;
}

Matrix::Matrix(int height, int width)
        : height(height), width(width), data(new float[height * width]) {}

Matrix::Matrix(const float* data, int height, int width)
        : data(new float[height * width]), height(height), width(width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            get(i, j) = (data + i * width)[j];
        }
    }
}

Matrix::Matrix(const Matrix& other) {
    height = other.getHeight();
    width = other.getWidth();
    data = new float[height * width];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            (data + i * width)[j] = other.get(i, j);
        }
    }
}

Matrix::Matrix(Matrix&& other) noexcept: height(other.height), width(other.width), data(other.data) {
    other.data = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) {
    height = other.height;
    width = other.width;
    data = other.data;
    other.data = nullptr;
    return *this;
}

Matrix::~Matrix() {
    delete[] data;
}

float& Matrix::get(int rowIndex, int colIndex) {
    return (data + rowIndex * width)[colIndex];
}

float Matrix::get(int rowIndex, int colIndex) const {
    return (data + rowIndex * width)[colIndex];
}

int Matrix::getWidth() const {
    return width;
}

int Matrix::getHeight() const {
    return height;
}

void Matrix::setNewDataWithSize(float* new_data, int new_height, int new_width) {
    delete[] data;
    data = new_data;
    height = new_height;
    width = new_width;
}

void Matrix::randomInit(int w) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> dist{0, 1};

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            get(i, j) = dist(gen) * sqrt(2.0 / w);
        }
    }
}

void Matrix::zeroInit() {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            get(i, j) = 0;
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "[";
    for (int i = 0; i < matrix.height; ++i) {
        os << "[";
        for (int j = 0; j < matrix.width; ++j) {
            os << matrix.get(i, j);
            if (j != matrix.width - 1) {
                os << " ";
            }
        }
        os << "]";
        if (i != matrix.height - 1) {
            os << "\n ";
        }
    }
    os << "]\n";
    return os;
}

float Matrix::sum() const {
    Matrix result = calculation[Config::getInstance().getProvider()]->sum(*this, -1);
    return result.get(0, 0);
}

Matrix Matrix::sum(const Matrix& matrix, int axis) {
    return calculation[Config::getInstance().getProvider()]->sum(matrix, axis);
}

Matrix Matrix::multiply(const Matrix& lhs, const Matrix& rhs) {
    return calculation[Config::getInstance().getProvider()]->multiply(lhs, rhs);
}

void Matrix::exp() {
    calculation[Config::getInstance().getProvider()]->exp_inline(*this);
}

Matrix Matrix::exp(const Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->exp(matrix);
}

void Matrix::log() {
    calculation[Config::getInstance().getProvider()]->log_inline(*this);
}

Matrix Matrix::log(const Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->log(matrix);
}

void Matrix::transpose() {
    calculation[Config::getInstance().getProvider()]->transpose_inline(*this);
}

Matrix Matrix::transpose(const Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->transpose(matrix);
}

void Matrix::clip(float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    calculation[Config::getInstance().getProvider()]->clip_inline(*this,
                                                                  minBound, maxBound,
                                                                  minValueToSet, maxValueToSet);
}

Matrix Matrix::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    return calculation[Config::getInstance().getProvider()]->clip(matrix,
                                                                  minBound, maxBound,
                                                                  minValueToSet, maxValueToSet);
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    return calculation[Config::getInstance().getProvider()]->sum(*this, rhs);
}

Matrix Matrix::operator+(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    return calculation[Config::getInstance().getProvider()]->sum(*this, matrix);
}

Matrix Matrix::operator-() const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = -1;
    return calculation[Config::getInstance().getProvider()]->elementWiseMultiply(*this, matrix);
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    return calculation[Config::getInstance().getProvider()]->subtract(*this, rhs);
}

Matrix Matrix::operator-(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    return calculation[Config::getInstance().getProvider()]->subtract(*this, matrix);
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    return calculation[Config::getInstance().getProvider()]->elementWiseMultiply(*this, rhs);
}

Matrix Matrix::operator*(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    return calculation[Config::getInstance().getProvider()]->elementWiseMultiply(*this, matrix);
}

Matrix Matrix::operator/(const Matrix& rhs) const {
    return calculation[Config::getInstance().getProvider()]->elementWiseDivide(*this, rhs);
}

Matrix Matrix::operator/(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    return calculation[Config::getInstance().getProvider()]->elementWiseDivide(*this, matrix);
}

float* Matrix::operator[](int index) {
    return data + index * width;
}

float* Matrix::operator[](int index) const {
    return data + index * width;
}

void Matrix::reciprocal() {
    calculation[Config::getInstance().getProvider()]->reciprocal_inline(*this);
}

Matrix Matrix::reciprocal(const Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->reciprocal(matrix);
}

Matrix::Ptr Matrix::merge(std::vector<Matrix::Ptr>::iterator begin,
                          std::vector<Matrix::Ptr>::iterator end,
                          int axis) {
    if (axis == 0) {
        int count = std::distance(begin, end);
        int newHeight = (*begin)->getHeight();
        int newWidth = (*begin)->getWidth() * count;
        Matrix::Ptr result(new Matrix(newHeight, newWidth));
        for (int i = 0; i < newHeight; ++i) {
            int currentWidth = 0;
            for (auto it = begin; it != end; ++it) {
                for (int j = 0; j < (*it)->getWidth(); ++j) {
                    result->get(i, currentWidth) = (*it)->get(i, j);
                    currentWidth++;
                }
            }
        }
        return result;
    }
    return {};
}

Matrix Matrix::argmax(const Matrix& matrix, int axis) {
    return calculation[Config::getInstance().getProvider()]->argmax(matrix, axis);
}