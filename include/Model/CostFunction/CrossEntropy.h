//
// Created by vlad on 4/26/23.
//

#ifndef CMAKE_AND_CUDA_CROSSENTROPY_H
#define CMAKE_AND_CUDA_CROSSENTROPY_H

#include "ICostFunction.h"

class CrossEntropy : public ICostFunction {
public:
    typedef std::unique_ptr<CrossEntropy> Ptr;

    float calculate(const Matrix& YHat, const Matrix& Y) override;

    Matrix derivative(const Matrix& YHat, const Matrix& Y) override;
};


#endif //CMAKE_AND_CUDA_CROSSENTROPY_H
