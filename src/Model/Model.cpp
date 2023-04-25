//
// Created by vlad on 4/24/23.
//

#include <algorithm>
#include <iostream>
#include "Model/Model.h"
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

void Model::train(int epochs, Matrix::Ptr train_x, Matrix::Ptr train_y, Matrix::Ptr val_x, Matrix::Ptr val_y) {
    for (int k = 0; k < epochs; ++k) {
        Matrix::Ptr current = train_x;
        for (const auto & layer : layers) {
            current = layer->forwardWithCache(current);
        }
        float cost = costFunction->calculate(*current, *train_y);
        Matrix::Ptr dCurrent = costFunction->derivative(*current, *train_y);

        for (int i = layers.size() - 1; i >= 0; i--) {
            dCurrent = layers[i]->backward(dCurrent, train_y->getWidth(), learningRate);
            layers[i]->clearCache();
        }
        std::cout << "Cost on " << k << " epoch: " << cost << "\n";
    }
}

Matrix::Ptr Model::predict(Matrix::Ptr input) {
    Matrix::Ptr current = input;
    for (const auto & layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

float Model::test(Matrix::Ptr test_x, Matrix::Ptr test_y) {
    Matrix::Ptr pred = predict(test_x);
    pred->clip(0.5, 0.5, 0, 1);
    int correct = 0;
    for (int i = 0; i < test_y->getWidth(); ++i) {
        if(abs(test_y->get(0, i) - pred->get(0, i)) < 0.1){
            correct++;
        }
    }
    return (float)correct / test_y->getWidth();
}