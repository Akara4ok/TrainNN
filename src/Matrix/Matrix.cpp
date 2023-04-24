//
// Created by vlad on 4/23/23.
//

#include <iostream>
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

Matrix::Matrix(int height, int width)
        : height(height), width(width), data(new float[height * width]) {}

Matrix::Matrix(Matrix &other) {
    height = other.getHeight();
    width = other.getWidth();
    data = new float[height * width];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            (data + i * width)[j] = other.get(i, j);
        }
    }
}

Matrix::~Matrix() {
    delete[] data;
}

float& Matrix::get(int index) {
    return (data + index)[0];
}
float& Matrix::get(int rowIndex, int colIndex) {
    return (data + rowIndex * width)[colIndex];
}

int Matrix::getWidth() const {
    return width;
}

int Matrix::getHeight() const {
    return height;
}

void Matrix::setNewDataWithSize(float *new_data, int new_height, int new_width) {
    delete[] data;
    data = new_data;
    height = new_height;
    width = new_width;
}

std::ostream &operator<<(std::ostream& os, Matrix& matrix) {
    os << "[";
    for (int i = 0; i < matrix.height; ++i) {
        os << "[";
        for (int j = 0; j < matrix.width; ++j) {
            os << matrix.get(i, j);
            if(j != matrix.width - 1){
                os << " ";
            }
        }
        os << "]";
        if(i != matrix.height - 1){
            os << "\n ";
        }
    }
    os << "]\n";
    return os;
}

float Matrix::sum() {
    Matrix::Ptr result = calculation[Config::getInstance().getProvider()]->sum(*this, -1);
    return result->get(0, 0);
}

Matrix::Ptr Matrix::sum(Matrix& matrix, int axis) {
    return calculation[Config::getInstance().getProvider()]->sum(matrix, axis);
}

Matrix::Ptr Matrix::multiply(Matrix& lhs, Matrix& rhs) {
    return calculation[Config::getInstance().getProvider()]->multiply(lhs, rhs);
}

void Matrix::exp() {
    calculation[Config::getInstance().getProvider()]->exp_inline(*this);
}

Matrix::Ptr Matrix::exp(Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->exp(matrix);
}

void Matrix::log() {
    calculation[Config::getInstance().getProvider()]->log_inline(*this);
}

Matrix::Ptr Matrix::log(Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->log(matrix);
}

void Matrix::transpose() {
    calculation[Config::getInstance().getProvider()]->transpose_inline(*this);
}

Matrix::Ptr Matrix::transpose(Matrix& matrix) {
    return calculation[Config::getInstance().getProvider()]->transpose(matrix);
}

Matrix::Ptr Matrix::elementWiseMultiply(Matrix& lhs, Matrix& rhs) {
    return calculation[Config::getInstance().getProvider()]->elementWiseMultiply(lhs, rhs);
}

void Matrix::clip(float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    calculation[Config::getInstance().getProvider()]->clip_inline(*this,
                                                                  minBound, maxBound,
                                                                  minValueToSet, maxValueToSet);
}

Matrix::Ptr Matrix::clip(Matrix &matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    return calculation[Config::getInstance().getProvider()]->clip(matrix,
                                                                  minBound, maxBound,
                                                                  minValueToSet, maxValueToSet);
}

Matrix::Ptr Matrix::operator+(Matrix &rhs) {
    return calculation[Config::getInstance().getProvider()]->sum(*this, rhs);
}

Matrix::Ptr Matrix::operator+(float value) {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    return calculation[Config::getInstance().getProvider()]->sum(*this, matrix);
}

Matrix::Ptr Matrix::operator-() {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = -1;
    return calculation[Config::getInstance().getProvider()]->elementWiseMultiply(*this, matrix);
}

Matrix::Ptr Matrix::operator-(Matrix &rhs) {
    return calculation[Config::getInstance().getProvider()]->subtract(*this, rhs);
}

void Matrix::reciprocal() {
    calculation[Config::getInstance().getProvider()]->reciprocal_inline(*this);
}

Matrix::Ptr Matrix::reciprocal(Matrix &matrix) {
    return calculation[Config::getInstance().getProvider()]->reciprocal(matrix);
}
