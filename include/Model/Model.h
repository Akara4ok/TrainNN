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
    float learningRate;
    Cost costType;
    ICostFunction::Ptr costFunction;
    std::vector<ILayer::Ptr> layers;
public:
    typedef std::unique_ptr<Model> Ptr;

    Model() = default;
    explicit Model(int inputSize);
    void add(ILayer::Ptr&& layer);
    void compile(float learningRate, Cost costType);
    void train(int epochs,
               const std::vector<Matrix::Ptr> &train_x, const std::vector<Matrix::Ptr> &train_y,
               const std::vector<Matrix::Ptr> &val_x, const std::vector<Matrix::Ptr> &val_y);
    Matrix::Ptr predict(Matrix::Ptr input);
    std::vector<Matrix::Ptr> predict(std::vector<Matrix::Ptr> input);
    void test(std::vector<Matrix::Ptr> test_x, std::vector<Matrix::Ptr> test_y);
    void serialize(std::string path);
    void deserialize(std::string path);
};


#endif //CMAKE_AND_CUDA_MODEL_H
