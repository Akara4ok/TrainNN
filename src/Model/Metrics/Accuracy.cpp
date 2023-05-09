//
// Created by vlad on 4/25/23.
//

#include "Model/Metrics/Accuracy.h"

float Accuracy::calculate(const Matrix& pred_y, const Matrix& test_y) {
    if (pred_y.getHeight() == 0) {
        return Accuracy::calculateBinary(pred_y, test_y);
    }
    return Accuracy::calculateMultiClass(pred_y, test_y);
}

float Accuracy::calculateBinary(const Matrix& pred_y, const Matrix& test_y) {
    Matrix diff = pred_y - test_y;
    Matrix clipped_diff = Matrix::clip(diff, -0.5 - THRESHOLD, 0.5 + THRESHOLD, 1, 1);
    Matrix isSameClass = Matrix::clip(clipped_diff, 0.5 + THRESHOLD, 0.5 + THRESHOLD, 1, 0);
    int correct = static_cast<int>(isSameClass.sum());
    return static_cast<float>(correct) / static_cast<float>(test_y.getWidth());
}

#include "iostream"
float Accuracy::calculateMultiClass(const Matrix& pred_y, const Matrix& test_y) {
    Matrix true_classes = Matrix::argmax(test_y, 1);
    Matrix pred_classes = Matrix::argmax(pred_y, 1);

    Matrix diff = true_classes - pred_classes;
    Matrix clipped_diff = Matrix::clip(diff, -THRESHOLD, THRESHOLD, 1, 1);
    Matrix isSameClass = Matrix::clip(clipped_diff, THRESHOLD, THRESHOLD, 1, 0);

    int correct = static_cast<int>(isSameClass.sum());

    return static_cast<float>(correct) / static_cast<float>(test_y.getWidth());
}
