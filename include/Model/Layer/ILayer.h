//
// Created by vlad on 4/24/23.
//

#ifndef CMAKE_AND_CUDA_ILAYER_H
#define CMAKE_AND_CUDA_ILAYER_H

#include "Matrix/Matrix.h"

class ILayer {
public:
    typedef std::unique_ptr<ILayer> Ptr;

    virtual int getHidden() = 0;
    virtual Matrix forward() = 0;
    virtual Matrix forwardWithCache() = 0;
    virtual Matrix backward() = 0;
    virtual void updateParams() = 0;
    virtual void initWeights(int previousHidden) = 0;
    virtual ~ILayer(){};
};


#endif //CMAKE_AND_CUDA_ILAYER_H
