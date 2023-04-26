//
// Created by vlad on 4/25/23.
//

#ifndef CMAKE_AND_CUDA_ACTIVATIONTYPES_H
#define CMAKE_AND_CUDA_ACTIVATIONTYPES_H

#include <fstream>
#include <string>

enum class Activation {
    Relu,
    Sigmoid,
    Softmax
};

std::ostream& operator<<(std::ostream& os, const Activation& other);

std::istream& operator>>(std::istream& is, Activation& type);

#endif //CMAKE_AND_CUDA_ACTIVATIONTYPES_H
