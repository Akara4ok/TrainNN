//
// Created by vlad on 4/25/23.
//

#include <math.h>
#include "Model/Metrics/Accuracy.h"

float Accuracy::calculate(Matrix& pred_y, Matrix& test_y) {
    if(pred_y.getHeight() == 0){
        return Accuracy::calculateBinary(pred_y, test_y);
    }
    return Accuracy::calculateMultiClass(pred_y, test_y);
}

float Accuracy::calculateBinary(Matrix& pred_y, Matrix& test_y) {
    Matrix::Ptr clipped_pred_y = Matrix::clip(pred_y, 0.5, 0.5, 0, 1);
    int correct = 0;
    for (int i = 0; i < test_y.getWidth(); ++i) {
        if (abs(test_y.get(0, i) - clipped_pred_y->get(0, i)) < 0.1) {
            correct++;
        }
    }
    return (float) correct / test_y.getWidth();
}

float Accuracy::calculateMultiClass(Matrix& pred_y, Matrix& test_y) {
    Matrix::Ptr true_classes = Matrix::argmax(test_y, 1);
    Matrix::Ptr pred_classes = Matrix::argmax(pred_y, 1);
    int correct = 0;
    for (int i = 0; i < true_classes->getWidth(); ++i) {
        if (abs(true_classes->get(0, i) - pred_classes->get(0, i)) < 0.1) {
            correct++;
        }
    }
    return (float) correct / test_y.getWidth();
}
