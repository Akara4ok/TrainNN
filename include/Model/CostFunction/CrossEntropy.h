//
// Created by vlad on 4/26/23.
//

#ifndef CMAKE_AND_CUDA_CROSSENTROPY_H
#define CMAKE_AND_CUDA_CROSSENTROPY_H

#include "ICostFunction.h"

class CrossEntropy : public ICostFunction{
public:
    typedef std::unique_ptr<CrossEntropy> Ptr;

    float calculate(Matrix& YHat, Matrix& Y) override;

    Matrix::Ptr derivative(Matrix& YHat, Matrix& Y) override;
};


#endif //CMAKE_AND_CUDA_CROSSENTROPY_H
