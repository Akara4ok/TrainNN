


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
    Config::getInstance().setProvider(Provider::GPU);
//    Provider provider = Config::getInstance().getProvider();
    int image_width = 32;
    int image_height = 32;
    ImageFlattenDataset::Ptr train_dataset =
            ImageFlattenDataset::createDataset("../Data/mnist/train",
                                               image_height, image_width,
                                               20, 500);

    auto train_x = train_dataset->getData();
    auto train_y = train_dataset->getLabel();
    ImageFlattenDataset::Ptr val_dataset =
            ImageFlattenDataset::createDataset("../Data/mnist/test",
                                               image_height, image_width,
                                               2000, 500);

    auto val_x = val_dataset->getData();
    auto val_y = val_dataset->getLabel();

    ImageFlattenDataset::Ptr test_dataset =
        ImageFlattenDataset::createDataset("../Data/mnistTest",
                                           image_height, image_width);

    auto test_x = test_dataset->getData();
    auto test_y = test_dataset->getLabel();

    std::cout << "Datasets have been read!" << std::endl;
    Model::Ptr model(new Model(image_height * image_width));
    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(70, Activation::Relu));
//    model->add(std::make_unique<Linear>(5, Activation::Relu));
    model->add(std::make_unique<Linear>(10, Activation::Softmax));
    model->compile(0.01, Cost::CrossEntropy);

    model->train(30, Verbose::All, train_x, train_y, val_x, val_y, "../Logs/v1");
//    model->deserialize("../Models/model.txt");

//    Matrix image = ImageFlattenDataset::preprocessImage("../Data/mnistTest/0/02.jpg",
//                                                        image_height, image_width);
//    Matrix pred_test_y = model->predict(image);
//    test_y[0]->copyCpuToGpu();
//    std::cout << *test_y[0];
//    std::cout << pred_test_y;
//    std::cout << Accuracy::calculate(pred_test_y, *test_y[0]);

//    model->serialize("../Models/model.txt");

//    model->serialize("../Models/model.txt");
//    auto pred = model->predict(x);
//    for (const auto& matrix : pred) {
//        std::cout << *matrix;
//    }
//    model->test(train_x, train_y);
    model->test(val_x, val_y);
}