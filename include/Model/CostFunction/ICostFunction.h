//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_ICOSTFUNCTION_H
#define CMAKE_AND_CUDA_ICOSTFUNCTION_H

#include <memory>
#include "Matrix/Matrix.h"

class ICostFunction {
public:
    typedef std::unique_ptr<ICostFunction> Ptr;

    virtual float calculate(Matrix& YHat, Matrix& Y) = 0;
    virtual Matrix::Ptr derivative(Matrix& YHat, Matrix& Y) = 0;
    virtual ~ICostFunction(){};
};

#endif //CMAKE_AND_CUDA_ICOSTFUNCTION_H
