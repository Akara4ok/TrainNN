//
// Created by vlad on 4/25/23.
//


#include "Model/Activation/ActivationTypes.h"

std::ostream& operator<<(std::ostream& os, const Activation& other) {
    switch (other) {
        case Activation::Relu:
            os << "Relu";
            break;
        case Activation::Sigmoid:
            os << "Sigmoid";
            break;
        case Activation::Softmax:
            os << "SoftmaxActivation";
            break;
        default:
            break;
    }
    return os;
};

std::istream& operator>>(std::istream& is, Activation& type) {
    std::string input;
    is >> input;
    if (input == "Relu") {
        type = Activation::Relu;
    } else if (input == "Sigmoid") {
        type = Activation::Sigmoid;
    } else if (input == "SoftmaxActivation") {
        type = Activation::Softmax;
    }
    return is;
};
