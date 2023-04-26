//
// Created by vlad on 4/25/23.
//

#ifndef CMAKE_AND_CUDA_COSTTYPES_H
#define CMAKE_AND_CUDA_COSTTYPES_H

#include <fstream>

enum class Cost {
    BinaryCrossEntropy,
    CrossEntropy
};

std::ostream& operator<<(std::ostream& os, const Cost& other);

std::istream& operator>>(std::istream& is, Cost& type);

#endif //CMAKE_AND_CUDA_COSTTYPES_H
