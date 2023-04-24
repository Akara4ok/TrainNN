//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_LINEAR_H
#define CMAKE_AND_CUDA_LINEAR_H

#include "Matrix/Matrix.h"
#include "ILayer.h"
#include "Model/Activation/IActivation.h"

class Linear : public ILayer{
    Matrix::Ptr W;
    Matrix::Ptr b;
    int hidden;
    IActivation::Ptr activation;
public:
    typedef std::unique_ptr<Linear> Ptr;

    Linear(int hidden, Activation type);

    int getHidden() override;
    Matrix forward() override;
    Matrix forwardWithCache() override;
    Matrix backward() override;
    void updateParams() override;
    void initWeights(int previousHidden) override;
};


#endif //CMAKE_AND_CUDA_LINEAR_H
