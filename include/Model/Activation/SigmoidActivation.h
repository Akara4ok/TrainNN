//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_SIGMOIDACTIVATION_H
#define CMAKE_AND_CUDA_SIGMOIDACTIVATION_H

#include "IActivation.h"

class SigmoidActivation : public IActivation {
public:
    typedef std::unique_ptr<SigmoidActivation> Ptr;

    Matrix::Ptr calculate(Matrix& matrix) override;

    Matrix::Ptr derivative(Matrix& X, Matrix& dA) override;
};

#endif //CMAKE_AND_CUDA_SIGMOIDACTIVATION_H