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

    Model::Ptr model(new Model(3200, 32));
    model->add(std::make_unique<Linear>(100, Activation::Relu));
    model->add(std::make_unique<Linear>(1000, Activation::Sigmoid));
    model->add(std::make_unique<Linear>(23, Activation::Relu));
    model->add(std::make_unique<Linear>(1, Activation::Sigmoid));
    model->compile(0.001, Cost::BinaryCrossEntropy);
}