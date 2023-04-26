//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <memory>
#include <algorithm>
#include "config.hpp"
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Dataset/ImageFlattenDataset.h"
#include "Model/Activation/SoftmaxActivation.h"
#include "Model/Metrics/Accuracy.h"

int main() {
    Config::getInstance().setProvider(Provider::CPU);
//    Provider provider = Config::getInstance().getProvider();
    int image_width = 32;
    int image_height = 32;
    ImageFlattenDataset::Ptr train_dataset(new ImageFlattenDataset("../Data/mnist/train",
                                                                   image_height, image_width,
                                                                   1024, 500));
    auto train_x = train_dataset->getData();
    auto train_y = train_dataset->getLabel();
//    std::cout << *train_y[0];
    ImageFlattenDataset::Ptr val_dataset(new ImageFlattenDataset("../Data/mnist/test",
                                                                 image_height, image_width,
                                                                 1024, 500));
    auto val_x = val_dataset->getData();
    auto val_y = val_dataset->getLabel();
    std::cout << "Datasets have been read!" << std::endl;
    Model::Ptr model(new Model(image_height * image_width));
    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(5, Activation::Relu));
    model->add(std::make_unique<Linear>(10, Activation::Softmax));
    model->compile(0.01, Cost::CrossEntropy);
//    model->serialize("../Models/model.txt");
//    model->deserialize("../Models/model.txt");
    model->train(500, train_x, train_y, val_x, val_y);
//    auto pred = model->predict(x);
//    for (const auto& matrix : pred) {
//        std::cout << *matrix;
//    }
//    std::cout << "!";
    model->test(train_x, train_y);
    model->test(val_x, val_y);
}