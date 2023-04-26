//
// Created by vlad on 4/24/23.
//
#include <iostream>
#include <fstream>

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

Matrix::Ptr Linear::forward(Matrix& input) {
    Matrix::Ptr x = Matrix::multiply(*W, input);
    x = *x + *b;
    x = activation->calculate(*x);
    return x;
}

Matrix::Ptr Linear::forwardWithCache(Matrix& input) {
    cache.push_back(Matrix::transpose(input));
    Matrix::Ptr x = Matrix::multiply(*W, input);
    x = *x + *b;
    cache.push_back(x);
    x = activation->calculate(*x);
    cache.push_back(x);
    return x;
}

Matrix::Ptr Linear::backward(Matrix& input, int m, float lr) {
    Matrix::Ptr dx = activation->derivative(*cache[1], input);
//    std::cout << *dx;
//    getchar();
    Matrix::Ptr dW = *Matrix::multiply(*dx, *cache[0]) / m;
    Matrix::Ptr db = *Matrix::sum(*dx, 0) / m;
    Matrix::Ptr output = Matrix::multiply(*Matrix::transpose(*W), *dx);
    updateParams(*dW, *db, lr);
    return output;
}

void Linear::updateParams(Matrix& dW, Matrix& db, float lr) {
    W = *W - *(dW * lr);
    b = *b - *(db * lr);
}

void Linear::createNewWeights(int previousHidden) {
    W = std::make_unique<Matrix>(hidden, previousHidden);
    b = std::make_unique<Matrix>(hidden, 1);
}

void Linear::initWeights(int previousHidden) {
    W = std::make_unique<Matrix>(hidden, previousHidden);
    b = std::make_unique<Matrix>(hidden, 1);
    W->randomInit(hidden, previousHidden);
    b->zeroInit();
}

void Linear::serialize(std::ofstream& file) {
    file << hidden << " " << W->getWidth() << " " << activationType << "\n";
    for (int i = 0; i < W->getHeight(); ++i) {
        for (int j = 0; j < W->getWidth(); ++j) {
            file << W->get(i, j) << " ";
        }
        file << b->get(i, 0) << "\n";
    }
}

void Linear::deserialize(std::ifstream& file) {
    int previousHidden;
    file >> hidden >> previousHidden >> activationType;
    W = std::make_unique<Matrix>(hidden, previousHidden);
    b = std::make_unique<Matrix>(hidden, 1);
    switch (activationType) {
        case Activation::Relu:
            activation = std::make_unique<ReluActivation>();
            break;
        case Activation::Sigmoid:
            activation = std::make_unique<SigmoidActivation>();
            break;
    }

    for (int i = 0; i < W->getHeight(); ++i) {
        for (int j = 0; j < W->getWidth(); ++j) {
            file >> W->get(i, j);
        }
        file >> b->get(i, 0);
    }
}
