//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_ILAYER_H
#define CMAKE_AND_CUDA_ILAYER_H

#include "Matrix/Matrix.h"

class ILayer {
public:
    typedef std::unique_ptr<ILayer> Ptr;

    virtual int getHidden() = 0;

    virtual void clearCache() = 0;

    virtual Matrix::Ptr forward(Matrix& input) = 0;

    virtual Matrix::Ptr forwardWithCache(Matrix& input) = 0;

    virtual Matrix::Ptr backward(Matrix&, int m, float lr) = 0;

    virtual void updateParams(Matrix& dW, Matrix& db, float lr) = 0;

    virtual void createNewWeights(int previousHidden) = 0;

    virtual void initWeights(int previousHidden) = 0;

    virtual void serialize(std::ofstream& file) = 0;

    virtual void deserialize(std::ifstream& file) = 0;

    virtual ~ILayer() = default;
};


#endif //CMAKE_AND_CUDA_ILAYER_H
