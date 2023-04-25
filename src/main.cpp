//
// Created by vlad on 4/23/23.
//

#include <iostream>
#include <memory>
#include "config.hpp"
#include "Matrix/Matrix.h"
#include "Model/CostFunction/BinaryCrossEntropy.h"
#include "Model/Model.h"
#include "Model/Layer/Linear.h"
#include "Dataset/ImageFlattenDataset.h"

#include <opencv2/opencv.hpp>

int main(){
    Config::getInstance().setProvider(Provider::CPU);
//    Provider provider = Config::getInstance().getProvider();
//    Matrix::Ptr matrix(new Matrix(5, 3));
//
//    for (int i = 0; i < matrix->getHeight(); ++i) {
//        for (int j = 0; j < matrix->getWidth(); ++j) {
//            matrix->get(i, j) = i * matrix->getWidth() + j;
//            if(i % 2 == 0){
//                matrix->get(i, j) = -matrix->get(i, j);
//            }
//        }
//    }
//
//    std::cout << (*matrix);
//    Matrix::Ptr y(new Matrix(1, 3));
//    Matrix::Ptr yhat(new Matrix(1, 3));
//    y->setNewDataWithSize(new float[3]{0, 0, 0}, 1, 3);
//    yhat->setNewDataWithSize(new float[3]{0.9, 0.9, 0.1}, 1, 3);
//
//
//    ICostFunction::Ptr entropy = BinaryCrossEntropy::Ptr(new BinaryCrossEntropy());
//    auto res = entropy->calculate(*yhat, *y);
//    std::cout << res;
//    Matrix::Ptr x(new Matrix(3200, 4));
//    x->randomInit();
//    Matrix::Ptr y(new Matrix(1, 4));
//    y->setNewDataWithSize(new float[4]{1, 1, 0, 0}, 1, 4);
//
//    Model::Ptr model(new Model(3200, 32));
//    model->add(std::make_unique<Linear>(100, Activation::Relu));
//    model->add(std::make_unique<Linear>(1000, Activation::Relu));
//    model->add(std::make_unique<Linear>(23, Activation::Relu));
//    model->add(std::make_unique<Linear>(1, Activation::Sigmoid));
//    model->compile(0.1, Cost::BinaryCrossEntropy);
//    model->train(100, x, y, x, y);
//    auto pred = model->predict(x);
//    std::cout << *pred;
//    std::cout << model->test(x, y);

//    cv::Mat image;
//    image = imread("../Data/PetImagesLite/Cat/0.jpg", cv::IMREAD_COLOR );
//    if ( !image.data )
//    {
//        printf("No image data \n");
//        return -1;
//    }
//    cv::imshow("Display Image", image);
//    cv::waitKey(0);
    ImageFlattenDataset::Ptr dataset(new ImageFlattenDataset("../Data/PetImagesLite",
                                                             64,
                                                             64, 42, 57));
    Matrix::Ptr x = dataset->getData()[0];
    Matrix::Ptr y = dataset->getLabel()[0];
    Model::Ptr model(new Model(64 * 64, 32));
    model->add(std::make_unique<Linear>(20, Activation::Relu));
    model->add(std::make_unique<Linear>(7, Activation::Relu));
    model->add(std::make_unique<Linear>(5, Activation::Relu));
    model->add(std::make_unique<Linear>(1, Activation::Sigmoid));
    model->compile(0.005, Cost::BinaryCrossEntropy);
    model->train(5000, x, y, x, y);
    auto pred = model->predict(x);
    std::cout << *pred;
    std::cout << model->test(x, y);
}