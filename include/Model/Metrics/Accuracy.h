//
// Created by vlad on 4/25/23.
//

#ifndef CMAKE_AND_CUDA_ACCURACY_H
#define CMAKE_AND_CUDA_ACCURACY_H

#include "Matrix/Matrix.h"

class Accuracy {
public:
    static float calculate(Matrix::Ptr pred_y, Matrix::Ptr test_y);
};


#endif //CMAKE_AND_CUDA_ACCURACY_H
