//
// Created by vlad on 4/24/23.
//

#include <algorithm>
#include <iostream>
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Model/CostFunction/BinaryCrossEntropy.h"
#include "Model/CostFunction/CrossEntropy.h"
#include "Model/Metrics/Accuracy.h"

Model::Model(int inputSize) : inputSize(inputSize) {}

void Model::add(ILayer::Ptr&& layer) {
    int previousNeurons = layers.empty() ? inputSize : layers.back()->getHidden();
    layer->initWeights(previousNeurons);
    layers.push_back(std::move(layer));
}

void Model::compile(float learningRate_, Cost costType_) {
    learningRate = learningRate_;
    switch (costType_) {
        case Cost::BinaryCrossEntropy:
            costType = Cost::BinaryCrossEntropy;
            costFunction = std::make_unique<BinaryCrossEntropy>();
            break;
        case Cost::CrossEntropy:
            costType = Cost::CrossEntropy;
            costFunction = std::make_unique<CrossEntropy>();
            break;
    }
}

void Model::train(int epochs,
                  const std::vector<Matrix::Ptr>& train_x, const std::vector<Matrix::Ptr>& train_y,
                  const std::vector<Matrix::Ptr>& val_x, const std::vector<Matrix::Ptr>& val_y) {
    for (int e = 0; e < epochs; ++e) {
        std::cout << "Epoch " << e << ": ";
        if (train_x.size() > 1) {
            std::cout << std::endl;
        }
        for (int t = 0; t < train_x.size(); ++t) {
            Matrix current(*train_x[t]);
            Matrix current_y(*train_y[t]);
            if (Config::getInstance().getProvider() == Provider::GPU) {
                current.copyCpuToGpu();
                current_y.copyCpuToGpu();
            }
            
            for (const auto& layer: layers) {
                current = layer->forwardWithCache(current);
            }
            float cost = costFunction->calculate(current, current_y);
            Matrix dCurrent = costFunction->derivative(current, current_y);
            for (int i = layers.size() - 1; i >= 0; i--) {
                dCurrent = layers[i]->backward(dCurrent, current_y.getWidth(), learningRate);
                layers[i]->clearCache();
            }
            if (train_x.size() > 1) {
                std::cout << "   Batch " << t << ": loss: " << cost << std::endl;
            } else {
                std::cout << "loss: " << cost << " ";
            }
        }

//        float val_loss = 0;
//        float val_accuracy = 0;
//        int datasetSize = 0;
//        for (int t = 0; t < val_x.size(); ++t) {
//            Matrix val_predict = predict(*val_x[t]);
//            datasetSize += val_x[t]->getWidth();
//            val_loss += costFunction->calculate(val_predict, *val_y[t]) * val_x[t]->getWidth();
//            val_accuracy += Accuracy::calculate(val_predict, *val_y[t]) * val_x[t]->getWidth();
//        }
//        std::cout << "-- val_loss: " << val_loss / datasetSize << " - ";
//        std::cout << "val_accuracy: " << val_accuracy / datasetSize << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Matrix Model::predict(const Matrix& input) {
    Matrix current = input;
    for (const auto& layer: layers) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<Matrix::Ptr> Model::predict(const std::vector<Matrix::Ptr>& input) {
    std::vector<Matrix::Ptr> result;
    for (const auto& matrix: input) {
        result.push_back(std::make_shared<Matrix>(predict(*matrix)));
    }
    return result;
}

void Model::test(std::vector<Matrix::Ptr> test_x, std::vector<Matrix::Ptr> test_y) {
    float test_loss = 0;
    float test_accuracy = 0;
    int datasetSize = 0;
    for (int t = 0; t < test_x.size(); ++t) {
        Matrix val_predict = predict(*test_x[t]);
        datasetSize += test_x[t]->getWidth();
        test_loss += costFunction->calculate(val_predict, *test_y[t]) * test_y[t]->getWidth();
        test_accuracy += Accuracy::calculate(val_predict, *test_y[t]) * test_y[t]->getWidth();
    }
    std::cout << "-- test_loss: " << test_loss / datasetSize << " - ";
    std::cout << "test_accuracy: " << test_accuracy / datasetSize << " ";
    std::cout << std::endl;
}

void Model::serialize(const std::string& path) {
    std::ofstream file(path);

    file << layers.size() << "\n";
    file << inputSize << "\n";
    file << learningRate << " " << costType << "\n";

    for (const auto& layer: layers) {
        file << "Linear\n";
        layer->serialize(file);
    }
    file.close();
}

void Model::deserialize(const std::string& path) {
    std::ifstream file(path);

    int numOfLayers;
    file >> numOfLayers;
    file >> inputSize;
    file >> learningRate >> costType;
    compile(learningRate, costType);

    for (int i = 0; i < numOfLayers; ++i) {
        std::string layerType;
        file >> layerType;
        layers.push_back(std::make_unique<Linear>());
        layers[i]->deserialize(file);
    }
}
