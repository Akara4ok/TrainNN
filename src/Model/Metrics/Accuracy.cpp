//
// Created by vlad on 4/25/23.
//

#include <math.h>
#include "Model/Metrics/Accuracy.h"

float Accuracy::calculate(Matrix::Ptr pred_y, Matrix::Ptr test_y) {
    pred_y = Matrix::clip(*pred_y, 0.5, 0.5, 0, 1);
    int correct = 0;
    for (int i = 0; i < test_y->getWidth(); ++i) {
        if(abs(test_y->get(0, i) - pred_y->get(0, i)) < 0.1){
            correct++;
        }
    }
    return (float)correct / test_y->getWidth();
}
