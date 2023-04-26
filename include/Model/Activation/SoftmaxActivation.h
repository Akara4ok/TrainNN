//
// Created by vlad on 4/26/23.
//

#ifndef TRAINNN_SOFTMAX_H
#define TRAINNN_SOFTMAX_H

#include "IActivation.h"

class SoftmaxActivation : public IActivation {
public:
    typedef std::unique_ptr<SoftmaxActivation> Ptr;

    Matrix::Ptr calculate(Matrix& matrix) override;

    Matrix::Ptr derivative(Matrix& X, Matrix& dA) override;
};


#endif //TRAINNN_SOFTMAX_H
