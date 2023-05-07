//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_LINEAR_H
#define CMAKE_AND_CUDA_LINEAR_H

#include <vector>
#include "Matrix/Matrix.h"
#include "ILayer.h"
#include "Model/Activation/ActivationTypes.h"
#include "Model/Activation/IActivation.h"

class Linear : public ILayer {
    Matrix W;
    Matrix b;
    std::vector<Matrix> cache; //A[l-1], x, A[l]
    int hidden{};
    Activation activationType{};
    IActivation::Ptr activation;
public:
    typedef std::unique_ptr<Linear> Ptr;

    Linear() = default;

    Linear(int hidden, Activation type);

    int getHidden() override;

    void clearCache() override;

    Matrix forward(const Matrix& input) override;

    Matrix forwardWithCache(const Matrix& input) override;

    Matrix backward(const Matrix& input, int m, float lr) override;

    void updateParams(const Matrix& dW, const Matrix& db, float lr) override;

    void createNewWeights(int previousHidden) override;

    void initWeights(int previousHidden) override;

    void serialize(std::ofstream& file) override;

    void deserialize(std::ifstream& file) override;
};


#endif //CMAKE_AND_CUDA_LINEAR_H
