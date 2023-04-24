//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_RELUACTIVATION_H
#define CMAKE_AND_CUDA_RELUACTIVATION_H

#include "IActivation.h"

class ReluActivation : public IActivation{
public:
    typedef std::unique_ptr<ReluActivation> Ptr;

    Matrix::Ptr calculate(Matrix& matrix) override;
    Matrix::Ptr derivative(Matrix &matrix) override;
};

#endif //CMAKE_AND_CUDA_RELUACTIVATION_H
