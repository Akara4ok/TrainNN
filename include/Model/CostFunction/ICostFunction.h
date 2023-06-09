//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_ICOSTFUNCTION_H
#define CMAKE_AND_CUDA_ICOSTFUNCTION_H

#include <memory>
#include "Matrix/Matrix.h"
#include "CostTypes.h"

class ICostFunction {
public:
    typedef std::unique_ptr<ICostFunction> Ptr;

    virtual float calculate(const Matrix& YHat, const Matrix& Y) = 0;

    virtual Matrix derivative(const Matrix& YHat, const Matrix& Y) = 0;

    virtual ~ICostFunction() = default;
};

#endif //CMAKE_AND_CUDA_ICOSTFUNCTION_H
