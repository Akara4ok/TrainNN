//
// Created by vlad on 4/25/23.
//

#ifndef CMAKE_AND_CUDA_ACCURACY_H
#define CMAKE_AND_CUDA_ACCURACY_H

#include "Matrix/Matrix.h"

class Accuracy {
    static float calculateBinary(Matrix& pred_y, Matrix& test_y);
    static float calculateMultiClass(Matrix& pred_y, Matrix& test_y);
public:
    static float calculate(Matrix& pred_y, Matrix& test_y);
};


#endif //CMAKE_AND_CUDA_ACCURACY_H
