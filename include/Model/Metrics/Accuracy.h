//
// Created by vlad on 4/25/23.
//

#ifndef CMAKE_AND_CUDA_ACCURACY_H
#define CMAKE_AND_CUDA_ACCURACY_H

#include "Matrix/Matrix.h"

class Accuracy {
    static float calculateBinary(const Matrix& pred_y, const Matrix& test_y);

    static float calculateMultiClass(const Matrix& pred_y, const Matrix& test_y);

public:
    static float calculate(const Matrix& pred_y, const Matrix& test_y);
};


#endif //CMAKE_AND_CUDA_ACCURACY_H
