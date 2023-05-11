//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <algorithm>
#include "config.hpp"
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Dataset/ImageFlattenDataset.h"
#include "Model/Activation/SoftmaxActivation.h"

int main() {
    Config::getInstance().setProvider(Provider::GPU);
//    Provider provider = Config::getInstance().getProvider();
    int imageWidth = 32;
    int imageHeight = 32;
    ImageFlattenDataset::Ptr trainDataset =
            ImageFlattenDataset::createDataset("../Data/mnist/train",
                                               imageHeight, imageWidth,
                                               20, 500);

    auto train_x = trainDataset->getData();
    auto train_y = trainDataset->getLabel();
    ImageFlattenDataset::Ptr valDataset =
            ImageFlattenDataset::createDataset("../Data/mnist/test",
                                               imageHeight, imageWidth,
                                               2000, 500);

    auto val_x = valDataset->getData();
    auto val_y = valDataset->getLabel();

    ImageFlattenDataset::Ptr testDataset =
            ImageFlattenDataset::createDataset("../Data/mnistTest",
                                               imageHeight, imageWidth);

    auto test_x = testDataset->getData();
    auto test_y = testDataset->getLabel();

    std::cout << "Datasets have been read!" << std::endl;

    Model::Ptr model(new Model({
                                       new Linear(70, Activation::Relu),
//                                       new Linear(70, Activation::Relu),
//                                       new Linear(70, Activation::Relu),
                                       new Linear(10, Activation::Softmax)
                               }, imageHeight * imageWidth));

    model->compile(0.01, Cost::CrossEntropy);

    model->train(30, Verbose::None, train_x, train_y, val_x, val_y, "../Logs/STANDARD");
//    model->deserialize("../Models/model.txt");

//    Matrix image = ImageFlattenDataset::preprocessImage("../Data/mnistTest/0/02.jpg",
//                                                        imageHeight, imageWidth);
//    Matrix pred_test_y = model->predict(image);
//    test_y[0]->copyCpuToGpu();
//    std::cout << *test_y[0];
//    std::cout << pred_test_y;
//    std::cout << Accuracy::calculate(pred_test_y, *test_y[0]);

//    model->serialize("../Models/model.txt");
    model->test(val_x, val_y);
}