//
// Created by vlad on 4/25/23.
//

#include <cmath>
#include "Model/Metrics/Accuracy.h"

float Accuracy::calculate(const Matrix& pred_y, const Matrix& test_y) {
    if (pred_y.getHeight() == 0) {
        return Accuracy::calculateBinary(pred_y, test_y);
    }
    return Accuracy::calculateMultiClass(pred_y, test_y);
}

float Accuracy::calculateBinary(const Matrix& pred_y, const Matrix& test_y) {
    Matrix clipped_pred_y = Matrix::clip(pred_y, 0.5, 0.5, 0, 1);
    if(Config::getInstance().getProvider() == Provider::GPU){
        clipped_pred_y.moveGpuToCpu();
    }
    int correct = 0;
    for (int i = 0; i < test_y.getWidth(); ++i) {
        if (std::abs(test_y[0][i] - clipped_pred_y[0][i]) < 0.1) {
            correct++;
        }
    }
    return (float) correct / test_y.getWidth();
}

float Accuracy::calculateMultiClass(const Matrix& pred_y, const Matrix& test_y) {
    Matrix true_classes = Matrix::argmax(test_y, 1);
    Matrix pred_classes = Matrix::argmax(pred_y, 1);
    if(Config::getInstance().getProvider() == Provider::GPU){
        true_classes.moveGpuToCpu();
        pred_classes.moveGpuToCpu();
    }
    int correct = 0;
    for (int i = 0; i < true_classes.getWidth(); ++i) {
        if (std::abs(true_classes[0][i] - pred_classes[0][i]) < 0.1) {
            correct++;
        }
    }
    return (float) correct / test_y.getWidth();
}
