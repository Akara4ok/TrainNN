//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_IACTIVATION_H
#define CMAKE_AND_CUDA_IACTIVATION_H

#include <memory>
#include "ActivationTypes.h"
#include "Matrix/Matrix.h"

class IActivation {
public:
    typedef std::unique_ptr<IActivation> Ptr;

    virtual Matrix::Ptr calculate(Matrix& matrix) = 0;

    virtual Matrix::Ptr derivative(Matrix& X, Matrix& dA) = 0;

    virtual ~IActivation() = default;
};

#endif //CMAKE_AND_CUDA_IACTIVATION_H
