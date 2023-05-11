//
// Created by vlad on 4/23/23.
//

#include "Matrix/Matrix.h"

#include <iostream>
#include "Cuda/CudaHelper.cuh"
#include "Matrix/Calculation/CpuMatrixCalculation.h"
#include "Matrix/Calculation/GpuMatrixCalculation.cuh"

std::map<Provider, std::shared_ptr<IMatrixCalculation>> Matrix::calculation;

void Matrix::initCalculation() {
    calculation.emplace(Provider::CPU, std::make_shared<CpuMatrixCalculation>());
    calculation.emplace(Provider::GPU, std::make_shared<GpuMatrixCalculation>());
}

static int initCalculationCaller = []() {
    Matrix::initCalculation();
    return 0;
}();

Provider Matrix::lastProvider = Provider::None;
std::shared_ptr<IMatrixCalculation> Matrix::currentAlgo;

Matrix::Matrix(int height, int width, Provider initProvider)
        : height(height), width(width) {
    isUseCpu = initProvider == Provider::CPU;
    isUseGpu = initProvider == Provider::GPU;
    if (isUseCpu) {
        data = new float[height * width];
    }

    if (lastProvider != Config::getInstance().getProvider()) {
        lastProvider = Config::getInstance().getProvider();
        currentAlgo = calculation[Config::getInstance().getProvider()];
    }
}

Matrix::Matrix(float* newData, int height, int width, Provider initProvider)
        : height(height), width(width) {
    isUseCpu = initProvider == Provider::CPU;
    isUseGpu = initProvider == Provider::GPU;
    if (!isUseGpu) {
        data = new float[height * width];
        std::copy(newData, newData + height * width, data);
    } else {
        gpuData = newData;
    }
}

Matrix::Matrix(const Matrix& other) {
    height = other.getHeight();
    width = other.getWidth();
    isUseCpu = other.isUseCpu;
    isUseGpu = other.isUseGpu;
    if (isUseCpu) {
        data = new float[height * width];
        std::copy(other.getData(), other.getData() + height * width, data);
    }
    if (isUseGpu) {
        CudaHelper::allocateGpuMemory(&gpuData, height * width);
        CudaHelper::copyFromGpuToGpu(other.gpuData, gpuData, height * width);
    }
}

Matrix::Matrix(Matrix&& other) noexcept: height(other.height), width(other.width),
                                         data(other.data), gpuData(other.gpuData),
                                         isUseCpu(other.isUseCpu), isUseGpu(other.isUseGpu) {
    other.data = nullptr;
    other.gpuData = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    height = other.height;
    width = other.width;
    if (isUseCpu) {
        delete[] data;
    }
    if (isUseGpu) {
        CudaHelper::deleteGpuMemory(gpuData);
    }
    data = other.data;
    gpuData = other.gpuData;
    isUseCpu = other.isUseCpu;
    isUseGpu = other.isUseGpu;
    other.data = nullptr;
    other.gpuData = nullptr;
    return *this;
}

Matrix::~Matrix() {
    if (isUseCpu) {
        delete[] data;
    }
    if (isUseGpu) {
        CudaHelper::deleteGpuMemory(gpuData);
    }
}

void Matrix::setNewDataWithSize(float* newData, int newHeight, int newWidth) {
    delete[] data;
    data = newData;
    height = newHeight;
    width = newWidth;
}

void Matrix::setNewGpuDataWithSize(float* newData, int newHeight, int newWidth) {
    setGpuData(newData);
    height = newHeight;
    width = newWidth;
}

void Matrix::setGpuData(float* newData) {
    if (gpuData != nullptr) {
        CudaHelper::deleteGpuMemory(gpuData);
    }
    gpuData = newData;
}

void Matrix::copyGpuToCpu() {
    isUseCpu = true;
    isUseCpu = true;
    delete[] data;
    data = new float[height * width];
    CudaHelper::copyFromGpuToCpu(gpuData, data, height * width);
}

void Matrix::copyCpuToGpu() {
    isUseGpu = true;
    if (gpuData != nullptr) {
        CudaHelper::deleteGpuMemory(gpuData);
    }
    CudaHelper::allocateGpuMemory(&gpuData, height * width);
    CudaHelper::copyFromCpuToGpu(data, gpuData, height * width);
}

Matrix Matrix::copy(const Matrix& other, Provider from, Provider to) {
    if (from == to)
        return {other};

    if (to == Provider::GPU) {
        float* gpuData;
        CudaHelper::allocateGpuMemory(&gpuData, other.getHeight() * other.getWidth());
        CudaHelper::copyFromCpuToGpu(other.getData(), gpuData, other.getHeight() * other.getWidth());
        return {gpuData, other.getHeight(), other.getWidth(), Provider::GPU};
    } else {
        auto* data = new float[other.getHeight() * other.getWidth()];
        CudaHelper::copyFromGpuToCpu(other.getGpuData(), data, other.getHeight() * other.getWidth());
        return {data, other.getHeight(), other.getWidth(), Provider::GPU};
    }
}

void Matrix::moveCpuToGpu() {
    isUseGpu = true;
    isUseCpu = false;
    if (gpuData != nullptr) {
        CudaHelper::deleteGpuMemory(gpuData);
    }
    CudaHelper::allocateGpuMemory(&gpuData, height * width);
    CudaHelper::copyFromCpuToGpu(data, gpuData, height * width);
    delete[] data;
    data = nullptr;
}

void Matrix::moveGpuToCpu() {
    isUseCpu = true;
    isUseCpu = true;
    delete[] data;
    data = new float[height * width];
    CudaHelper::copyFromGpuToCpu(gpuData, data, height * width);
    CudaHelper::deleteGpuMemory(gpuData);
    gpuData = nullptr;
}

void Matrix::randomInit(int w) {
    currentAlgo->randomInit(*this, w);
}

void Matrix::zeroInit() {
    if (isUseCpu) {
        calculation[Provider::CPU]->zeroInit(*this);
    }
    if (isUseGpu) {
        calculation[Provider::GPU]->zeroInit(*this);
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
    Matrix result = currentAlgo->sum(*this, -1);
    if (Config::getInstance().getProvider() == Provider::GPU) {
        result.copyGpuToCpu();
    }
    return result.get(0, 0);
}

Matrix Matrix::sum(const Matrix& matrix, int axis) {
    return currentAlgo->sum(matrix, axis);
}

Matrix Matrix::multiply(const Matrix& lhs, const Matrix& rhs) {
    return currentAlgo->multiply(lhs, rhs);
}

void Matrix::exp() {
    currentAlgo->expInline(*this);
}

Matrix Matrix::exp(const Matrix& matrix) {
    return currentAlgo->exp(matrix);
}

void Matrix::log() {
    currentAlgo->logInline(*this);
}

Matrix Matrix::log(const Matrix& matrix) {
    return currentAlgo->log(matrix);
}

void Matrix::transpose() {
    currentAlgo->transposeInline(*this);
}

Matrix Matrix::transpose(const Matrix& matrix) {
    return currentAlgo->transpose(matrix);
}

void Matrix::clip(float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    currentAlgo->clipInline(*this,
                            minBound, maxBound,
                            minValueToSet, maxValueToSet);
}

Matrix Matrix::clip(const Matrix& matrix, float minBound, float maxBound, float minValueToSet, float maxValueToSet) {
    return currentAlgo->clip(matrix,
                             minBound, maxBound,
                             minValueToSet, maxValueToSet);
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    return currentAlgo->sum(*this, rhs);
}

Matrix Matrix::operator+(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        matrix.moveCpuToGpu();
    }
    return currentAlgo->sum(*this, matrix);
}

Matrix Matrix::operator-() const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = -1;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        matrix.moveCpuToGpu();
    }
    return currentAlgo->elementWiseMultiply(*this, matrix);
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    return currentAlgo->subtract(*this, rhs);
}

Matrix Matrix::operator-(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        matrix.moveCpuToGpu();
    }
    return currentAlgo->subtract(*this, matrix);
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    return currentAlgo->elementWiseMultiply(*this, rhs);
}

Matrix Matrix::operator*(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        matrix.moveCpuToGpu();
    }
    return currentAlgo->elementWiseMultiply(*this, matrix);
}

Matrix Matrix::operator/(const Matrix& rhs) const {
    return currentAlgo->elementWiseDivide(*this, rhs);
}

Matrix Matrix::operator/(float value) const {
    Matrix matrix(1, 1);
    matrix.get(0, 0) = value;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        matrix.moveCpuToGpu();
    }
    return currentAlgo->elementWiseDivide(*this, matrix);
}

void Matrix::reciprocal() {
    currentAlgo->reciprocalInline(*this);
}

Matrix Matrix::reciprocal(const Matrix& matrix) {
    return currentAlgo->reciprocal(matrix);
}

Matrix::Ptr Matrix::merge(std::vector<Matrix::Ptr>::iterator begin,
                          std::vector<Matrix::Ptr>::iterator end,
                          int axis) {
    if (axis == 0) {
        int count = static_cast<int>(std::distance(begin, end));
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
    return currentAlgo->argmax(matrix, axis);
}