//
// Created by vlad on 4/24/23.
//

#include "Model/Model.h"
#include <algorithm>
#include "Model/CostFunction/BinaryCrossEntropy.h"

Model::Model() {}

Model::Model(int inputSize, int batchSize) : inputSize(inputSize), batchSize(batchSize) {}

void Model::add(ILayer::Ptr&& layer) {
    int previousNeurons = layers.size() == 0 ? inputSize : layers.back()->getHidden();
    layer->initWeights(previousNeurons);
    layers.push_back(std::move(layer));
}

void Model::compile(float learningRate, Cost costType) {
    this->learningRate = learningRate;
    switch (costType) {
        case Cost::BinaryCrossEntropy:
            costFunction = std::make_unique<BinaryCrossEntropy>();
            break;
    }
}

void Model::train(int epochs, Matrix &train_x, Matrix &train_y, Matrix &val_x, Matrix &val_y) {

}

Matrix::Ptr Model::predict(Matrix &input) {
    return Matrix::Ptr();
}

void Model::test(Matrix &test_x, Matrix &test_y) {

}
