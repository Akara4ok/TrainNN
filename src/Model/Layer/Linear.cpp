//
// Created by vlad on 4/24/23.
//
#include <iostream>

#include "Model/Layer/Linear.h"
#include "Model/Activation/SigmoidActivation.h"
#include "Model/Activation/ReluActivation.h"
#include "Model/Activation/SoftmaxActivation.h"

Linear::Linear(int hidden, Activation type) : hidden(hidden), activationType(type) {
    switch (type) {
        case Activation::Relu:
            activation = std::make_unique<ReluActivation>();
            break;
        case Activation::Sigmoid:
            activation = std::make_unique<SigmoidActivation>();
            break;
        case Activation::Softmax:
            activation = std::make_unique<SoftmaxActivation>();
            break;
    }
}

int Linear::getHidden() {
    return hidden;
}

void Linear::clearCache() {
    cache.clear();
}

Matrix Linear::forward(const Matrix& input) {
    Matrix x = Matrix::multiply(W, input);
    x = x + b;
    x = activation->calculate(x);
    return x;
}

Matrix Linear::forwardWithCache(const Matrix& input) {
    cache.push_back(Matrix::transpose(input));
    W.copyGpuToCpu();
    b.copyGpuToCpu();
    Matrix x = input;
    x.copyGpuToCpu();
    x = Matrix::multiply(W, input);
    x.copyGpuToCpu();
    for (int i = 0; i < 20; ++i) {
        float k = x[i][0];
        k += 1;
    }
    x = x + b;
    x.copyGpuToCpu();
    for (int i = 0; i < 20; ++i) {
        float k = x[i][0];
        k += 1;
    }
    cache.push_back(x);
//    if(hidden == 10){
//        std::cout << x;
//    }
    x = activation->calculate(x);
    x.copyGpuToCpu();
    cache.push_back(x);
    return x;
}

Matrix Linear::backward(const Matrix& input, int m, float lr) {
    Matrix dx = activation->derivative(cache[1], input);
    dx.copyGpuToCpu();
//    if(hidden == 10){
//        std::cout << dx;
//    }
    Matrix dW = Matrix::multiply(dx, cache[0]) / m;
    dW.copyGpuToCpu();
    Matrix db = Matrix::sum(dx, 0) / m;
    Matrix output = Matrix::multiply(Matrix::transpose(W), dx);
    updateParams(dW, db, lr);
    W.copyGpuToCpu();
    return output;
}

void Linear::updateParams(const Matrix& dW, const Matrix& db, float lr) {
//    W.copyGpuToCpu();
    W = W - (dW * lr);
//    W.copyGpuToCpu();
    b = b - (db * lr);
}

void Linear::createNewWeights(int previousHidden) {
    W = Matrix(hidden, previousHidden, Config::getInstance().getProvider());
    b = Matrix(hidden, 1, Config::getInstance().getProvider());
}

void Linear::initWeights(int previousHidden) {
    W = Matrix(hidden, previousHidden, Config::getInstance().getProvider());
    b = Matrix(hidden, 1, Config::getInstance().getProvider());
    W.randomInit(previousHidden);
    b.zeroInit();
}

void Linear::serialize(std::ofstream& file) {
    file << hidden << " " << W.getWidth() << " " << activationType << "\n";
    for (int i = 0; i < W.getHeight(); ++i) {
        for (int j = 0; j < W.getWidth(); ++j) {
            file << W[i][j] << " ";
        }
        file << b[i][0] << "\n";
    }
}

void Linear::deserialize(std::ifstream& file) {
    int previousHidden;
    file >> hidden >> previousHidden >> activationType;
    W = Matrix(hidden, previousHidden);
    b = Matrix(hidden, 1);
    switch (activationType) {
        case Activation::Relu:
            activation = std::make_unique<ReluActivation>();
            break;
        case Activation::Sigmoid:
            activation = std::make_unique<SigmoidActivation>();
            break;
        case Activation::Softmax:
            activation = std::make_unique<SoftmaxActivation>();
            break;
    }

    for (int i = 0; i < W.getHeight(); ++i) {
        for (int j = 0; j < W.getWidth(); ++j) {
            file >> W[i][j];
        }
        file >> b[i][0];
    }
}
