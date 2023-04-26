//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_BINARYCROSSENTROPY_H
#define CMAKE_AND_CUDA_BINARYCROSSENTROPY_H

#include "ICostFunction.h"

class BinaryCrossEntropy : public ICostFunction {
public:
    typedef std::unique_ptr<BinaryCrossEntropy> Ptr;

    float calculate(Matrix& YHat, Matrix& Y) override;

    Matrix::Ptr derivative(Matrix& YHat, Matrix& Y) override;
};

#endif //CMAKE_AND_CUDA_BINARYCROSSENTROPY_H
