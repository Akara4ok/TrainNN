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
    Matrix clippedDiff = Matrix::clip(diff, -0.5 - THRESHOLD, 0.5 + THRESHOLD, 1, 1);
    Matrix isSameClass = Matrix::clip(clippedDiff, 0.5 + THRESHOLD, 0.5 + THRESHOLD, 1, 0);
    int correct = static_cast<int>(isSameClass.sum());
    return static_cast<float>(correct) / static_cast<float>(test_y.getWidth());
}

float Accuracy::calculateMultiClass(const Matrix& pred_y, const Matrix& test_y) {
    Matrix trueClasses = Matrix::argmax(test_y, 1);
    Matrix predClasses = Matrix::argmax(pred_y, 1);

    Matrix diff = trueClasses - predClasses;
    Matrix clippedDiff = Matrix::clip(diff, -THRESHOLD, THRESHOLD, 1, 1);
    Matrix isSameClass = Matrix::clip(clippedDiff, THRESHOLD, THRESHOLD, 1, 0);

    int correct = static_cast<int>(isSameClass.sum());

    return static_cast<float>(correct) / static_cast<float>(test_y.getWidth());
}
