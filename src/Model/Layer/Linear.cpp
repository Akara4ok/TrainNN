//
// Created by vlad on 4/24/23.
//

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

Matrix Linear::forward() {
    return Matrix(0, 0);
}

Matrix Linear::forwardWithCache() {
    return Matrix(0, 0);
}

Matrix Linear::backward() {
    return Matrix(0, 0);
}

void Linear::updateParams() {

}

void Linear::initWeights(int previousHidden) {
    W = std::make_unique<Matrix>(hidden, previousHidden);
    b = std::make_unique<Matrix>(hidden, 1);
}
