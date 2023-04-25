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

int main(){
    Config::getInstance().setProvider(Provider::CPU);
//    Provider provider = Config::getInstance().getProvider();
    ImageFlattenDataset::Ptr train_dataset(new ImageFlattenDataset("../Data/PetImages/train",
                                                                 64,64,
                                                                 1024, 500));
    auto train_x = train_dataset->getData();
    auto train_y = train_dataset->getLabel();
    ImageFlattenDataset::Ptr val_dataset(new ImageFlattenDataset("../Data/PetImages/test",
                                                                   64,64,
                                                                   1024, 50));
    auto val_x = val_dataset->getData();
    auto val_y = val_dataset->getLabel();
    std::cout << "Datasets have been read!" << std::endl;
    Model::Ptr model(new Model(64 * 64));
    model->add(std::make_unique<Linear>(20, Activation::Relu));
    model->add(std::make_unique<Linear>(7, Activation::Relu));
    model->add(std::make_unique<Linear>(5, Activation::Relu));
    model->add(std::make_unique<Linear>(1, Activation::Sigmoid));
    model->compile(0.005, Cost::BinaryCrossEntropy);
//    model->serialize("../Models/model.txt");
//    model->deserialize("../Models/model.txt");
    model->train(5, train_x, train_y, val_x, val_y);
//    auto pred = model->predict(x);
//    for (const auto& matrix : pred) {
//        std::cout << *matrix;
//    }
//    std::cout << "!";
    model->test(val_x, val_y);
}