//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_MODEL_H
#define CMAKE_AND_CUDA_MODEL_H

#include <vector>
#include <string>
#include "Matrix/Matrix.h"
#include "Model/Layer/ILayer.h"
#include "Model/CostFunction/ICostFunction.h"
#include "Monitoring/Monitoring.h"

class Model {
    int inputSize{};
    float learningRate{};
    Cost costType{};
    ICostFunction::Ptr costFunction;
    std::vector<ILayer::Ptr> layers;
    int numberOfParameters = 0;
public:
    typedef std::unique_ptr<Model> Ptr;

    Model() = default;

    explicit Model(int inputSize);

    explicit Model(std::initializer_list<ILayer*> layers, int inputSize);

    void add(ILayer* layer);

    void add(ILayer::Ptr&& layer);

    [[nodiscard]] int getNumberOfParams() const;

    void compile(float learningRate_, Cost costType_);

    void
    train(int epochs, Verbose verb, const std::vector<Matrix::Ptr>& train_x, const std::vector<Matrix::Ptr>& train_y,
          const std::vector<Matrix::Ptr>& val_x, const std::vector<Matrix::Ptr>& val_y,
          const std::string& logFolder = "");

    Matrix predict(const Matrix& input);

    std::vector<Matrix::Ptr> predict(const std::vector<Matrix::Ptr>& input);

    void test(std::vector<Matrix::Ptr> test_x, std::vector<Matrix::Ptr> test_y);

    void serialize(const std::string& path);

    void deserialize(const std::string& path);
};


#endif //CMAKE_AND_CUDA_MODEL_H
