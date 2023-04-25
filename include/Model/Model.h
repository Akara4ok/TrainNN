//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_MODEL_H
#define CMAKE_AND_CUDA_MODEL_H

#include <vector>
#include "Matrix/Matrix.h"
#include "Model/Layer/ILayer.h"
#include "Model/CostFunction/ICostFunction.h"

class Model {
    int inputSize;
    int batchSize;
    float learningRate;
    ICostFunction::Ptr costFunction;
    std::vector<ILayer::Ptr> layers;
public:
    typedef std::unique_ptr<Model> Ptr;

    Model();
    Model(int inputSize, int batchSize);
    void add(ILayer::Ptr&& layer);
    void compile(float learningRate, Cost costType);
    void train(int epochs, Matrix::Ptr train_x, Matrix::Ptr train_y, Matrix::Ptr val_x, Matrix::Ptr val_y);
    Matrix::Ptr predict(Matrix::Ptr input);
    float test(Matrix::Ptr test_x, Matrix::Ptr test_y);
};


#endif //CMAKE_AND_CUDA_MODEL_H
