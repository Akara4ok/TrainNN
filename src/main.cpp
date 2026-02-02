//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <algorithm>
#include "config.hpp"
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Dataset/ImageFlattenDataset.h"
#include "Model/Metrics/Accuracy.h"
#include "Model/Activation/SoftmaxActivation.h"

void testModelTraining(const std::string &dataDir, int imageHeight, int imageWidth) {
    ImageFlattenDataset::Ptr trainDataset =
        ImageFlattenDataset::createDataset(dataDir + "/train",
                                           imageHeight, imageWidth,
                                           200, 2000);

    auto train_x = trainDataset->getData();
    auto train_y = trainDataset->getLabel();
    ImageFlattenDataset::Ptr valDataset =
        ImageFlattenDataset::createDataset(dataDir + "/test",
                                           imageHeight, imageWidth,
                                           2000, 500);

    auto val_x = valDataset->getData();
    auto val_y = valDataset->getLabel();

    std::cout << "Datasets have been read!" << std::endl;

    Model::Ptr model(new Model({new Linear(70, Activation::Relu),
                                new Linear(10, Activation::Softmax)},
                               imageHeight * imageWidth));

    model->compile(0.01, Cost::CrossEntropy);

    model->train(30, Verbose::None, train_x, train_y, val_x, val_y, "Logs/STANDARD");

    model->serialize("Models/model.txt");
    model->test(val_x, val_y);
}

void testModelInference(const std::string &dataDir, int imageHeight, int imageWidth) {
    ImageFlattenDataset::Ptr testDataset =
        ImageFlattenDataset::createDataset(dataDir,
                                           imageHeight, imageWidth);

    auto test_x = testDataset->getData();
    auto test_y = testDataset->getLabel();

    std::cout << "Datasets have been read!" << std::endl;

    Model::Ptr model(new Model({new Linear(70, Activation::Relu),
                                new Linear(10, Activation::Softmax)},
                               imageHeight * imageWidth));

    model->deserialize("Models/model.txt");

    Matrix image = ImageFlattenDataset::preprocessImage(dataDir + "/0/0.jpg",
                                                        imageHeight, imageWidth);
    Matrix pred_test_y = model->predict(image);

    if(test_y.empty()){
        std::cerr << "Empty test data" << std::endl;
        return;
    }

    test_y[0]->copyCpuToGpu();
    std::cout << *test_y[0];
    std::cout << pred_test_y;
} 

int main()
{
    Config::getInstance().setProvider(Provider::GPU);

    int imageWidth = 32;
    int imageHeight = 32;

    testModelTraining("Data/mnist", imageHeight, imageWidth);
    // testModelInference("Data/mnist_test", imageHeight, imageWidth);
}