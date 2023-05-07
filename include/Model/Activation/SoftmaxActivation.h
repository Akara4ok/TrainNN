//
// Created by vlad on 4/26/23.
//

#ifndef TRAINNN_SOFTMAX_H
#define TRAINNN_SOFTMAX_H

#include "IActivation.h"

class SoftmaxActivation : public IActivation {
public:
    typedef std::unique_ptr<SoftmaxActivation> Ptr;

    Matrix calculate(const Matrix& matrix) override;

    Matrix derivative(const Matrix& X, const Matrix& dA) override;
};


#endif //TRAINNN_SOFTMAX_H
