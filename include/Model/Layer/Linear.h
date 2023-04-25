//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_LINEAR_H
#define CMAKE_AND_CUDA_LINEAR_H

#include <vector>
#include "Matrix/Matrix.h"
#include "ILayer.h"
#include "Model/Activation/IActivation.h"

class Linear : public ILayer{
    Matrix::Ptr W;
    Matrix::Ptr b;
    std::vector<Matrix::Ptr> cache; //A[l-1], x, A[l]
    int hidden;
    IActivation::Ptr activation;
public:
    typedef std::unique_ptr<Linear> Ptr;

    Linear(int hidden, Activation type);

    int getHidden() override;
    void clearCache() override;
    Matrix::Ptr forward(Matrix::Ptr input) override;
    Matrix::Ptr forwardWithCache(Matrix::Ptr input) override;
    Matrix::Ptr backward(Matrix::Ptr input, int m, float lr) override;
    void updateParams(Matrix::Ptr dW, Matrix::Ptr db, float lr) override;
    void initWeights(int previousHidden) override;
};


#endif //CMAKE_AND_CUDA_LINEAR_H
