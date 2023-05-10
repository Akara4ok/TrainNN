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

Model::Model(int inputSize) : inputSize(inputSize) {

}

Model::Model(std::initializer_list<ILayer*> layersPtr, int inputSize) : inputSize(inputSize) {
    for (auto layer: layersPtr) {
        add(layer);
    }
}

void Model::add(ILayer* layer) {
    int previousNeurons = layers.empty() ? inputSize : layers.back()->getHidden();
    numberOfParameters += previousNeurons * layer->getHidden() + layer->getHidden();
    layer->initWeights(previousNeurons);
    layers.emplace_back(layer);
}

void Model::add(ILayer::Ptr&& layer) {
    int previousNeurons = layers.empty() ? inputSize : layers.back()->getHidden();
    numberOfParameters += previousNeurons * layer->getHidden() + layer->getHidden();
    layer->initWeights(previousNeurons);
    layers.push_back(std::move(layer));
}

int Model::getNumberOfParams() const {
    return numberOfParameters;
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

void
Model::train(int epochs, Verbose verb, const std::vector<Matrix::Ptr>& train_x, const std::vector<Matrix::Ptr>& train_y,
             const std::vector<Matrix::Ptr>& val_x, const std::vector<Matrix::Ptr>& val_y,
             const std::string& logFolder) {
    Monitoring monitoring(train_x[0]->getWidth(), static_cast<int>(train_x.size()), numberOfParameters, verb);
    for (int e = 0; e < epochs; ++e) {
        for (int t = 0; t < train_x.size(); ++t) {
            Matrix current;
            Matrix current_y;
            if (Config::getInstance().getProvider() == Provider::GPU) {
                current = Matrix::copyCpuToGpu(*train_x[t]);
                current_y = Matrix::copyCpuToGpu(*train_y[t]);
            } else {
                current = Matrix(*train_x[t]);
                current_y = Matrix(*train_y[t]);
            }

            for (const auto& layer: layers) {
                current = layer->forwardWithCache(current);
            }
            float cost = costFunction->calculate(current, current_y);
            Matrix dCurrent = costFunction->derivative(current, current_y);
            for (int i = static_cast<int>(layers.size() - 1); i >= 0; i--) {
                dCurrent = layers[i]->backward(dCurrent, current_y.getWidth(), learningRate);
                layers[i]->clearCache();
            }
            monitoring.add(e, t, cost);
            monitoring.logLastSample();
        }

        float val_loss = 0;
        float val_accuracy = 0;
        int datasetSize = 0;
        for (int t = 0; t < val_x.size(); ++t) {
            Matrix current_val_y;
            if (Config::getInstance().getProvider() == Provider::GPU) {
                current_val_y = Matrix::copyCpuToGpu(*val_y[t]);
            } else {
                current_val_y = Matrix(*val_y[t]);
            }
            Matrix val_predict = predict(*val_x[t]);
            datasetSize += val_x[t]->getWidth();
            val_loss += costFunction->calculate(val_predict, current_val_y) * static_cast<float>(val_x[t]->getWidth());
            val_accuracy += Accuracy::calculate(val_predict, current_val_y) * static_cast<float>(val_x[t]->getWidth());
        }
        val_loss /= static_cast<float>(datasetSize);
        val_accuracy /= static_cast<float>(datasetSize);
        monitoring.add(e, -1, std::numeric_limits<float>::lowest(), val_loss, val_accuracy);
        monitoring.logLastSample();
    }
    if (!logFolder.empty()) {
        monitoring.serialize(logFolder);
    }
}

Matrix Model::predict(const Matrix& input) {
    Matrix current;
    if (Config::getInstance().getProvider() == Provider::GPU) {
        current = Matrix::copyCpuToGpu(input);
    } else {
        current = Matrix(input);
    }
    for (const auto& layer: layers) {
        current = layer->forward(current);
    }
    if (Config::getInstance().getProvider() == Provider::GPU) {
        current.copyGpuToCpu();
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
        Matrix current_test_y(*test_y[t]);
        if (Config::getInstance().getProvider() == Provider::GPU) {
            current_test_y.copyCpuToGpu();
        }
        datasetSize += test_x[t]->getWidth();
        test_loss += costFunction->calculate(val_predict, current_test_y) * static_cast<float>(test_y[t]->getWidth());
        test_accuracy += Accuracy::calculate(val_predict, current_test_y) * static_cast<float>(test_y[t]->getWidth());
    }
    std::cout << "Test: ";
    std::cout << "test_loss: " << test_loss / static_cast<float>(datasetSize) << " - ";
    std::cout << "test_accuracy: " << test_accuracy / static_cast<float>(datasetSize) << " ";
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

    layers.clear();

    for (int i = 0; i < numOfLayers; ++i) {
        std::string layerType;
        file >> layerType;
        layers.push_back(std::make_unique<Linear>());
        layers[i]->deserialize(file);
    }
}
