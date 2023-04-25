//
// Created by vlad on 4/24/23.
//
#include <iostream>

#include "Model/Layer/Linear.h"
#include "Model/Activation/SigmoidActivation.h"
#include "Model/Activation/ReluActivation.h"

Linear::Linear(int hidden, Activation type) : hidden(hidden) {
    switch (type) {
        case Activation::Relu:
            activation = std::make_unique<ReluActivation>();
            break;
        case Activation::Sigmoid:
            activation = std::make_unique<SigmoidActivation>();
            break;
    }
}

int Linear::getHidden() {
    return hidden;
}

void Linear::clearCache() {
    cache.clear();
}

Matrix::Ptr Linear::forward(Matrix::Ptr input) {
    Matrix::Ptr x = Matrix::multiply(*W, *input);
    x = *x + *b;
    x = activation->calculate(*x);
    return x;
}

Matrix::Ptr Linear::forwardWithCache(Matrix::Ptr input) {
    cache.push_back(Matrix::transpose(*input));
    Matrix::Ptr x = Matrix::multiply(*W, *input);
    x = *x + *b;
    cache.push_back(x);
    x = activation->calculate(*x);
    cache.push_back(x);
    return x;
}

Matrix::Ptr Linear::backward(Matrix::Ptr input, int m, float lr) {
    Matrix::Ptr dx = *input * *activation->derivative(*cache[1]);
    Matrix::Ptr dW = *Matrix::multiply(*dx, *cache[0]) / m;
    Matrix::Ptr db = *Matrix::sum(*dx, 0) / m;
    Matrix::Ptr output = Matrix::multiply(*Matrix::transpose(*W), *dx);
    updateParams(dW, db, lr);
    return output;
}

void Linear::updateParams(Matrix::Ptr dW, Matrix::Ptr db, float lr) {
    W = *W - *(*dW * lr);
    b = *b - *(*db * lr);
}

void Linear::initWeights(int previousHidden) {
    W = std::make_unique<Matrix>(hidden, previousHidden);
    b = std::make_unique<Matrix>(hidden, 1);
    W->randomInit(hidden, previousHidden);
    //W = *W * 0.01;
    b->zeroInit();
}
