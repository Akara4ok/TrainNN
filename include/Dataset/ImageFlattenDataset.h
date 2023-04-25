//
// Created by vlad on 4/25/23.
//

#ifndef TRAINNN_IMAGEFLATTENDATASET_H
#define TRAINNN_IMAGEFLATTENDATASET_H

#include "Dataset.h"

class ImageFlattenDataset : public Dataset {
    int imageHeight;
    int imageWidth;
    std::vector<std::string> imagePaths;
public:
    ImageFlattenDataset(std::string folderPath, int imageHeight, int imageWidth, int batchSize = -1,
                        int fixedSize = -1, int seed = 42);
    Matrix::Ptr preprocessImage(std::string imagePath) override;
    Matrix::Ptr preprocessLabel(std::string imagePath) override;
};


#endif //TRAINNN_IMAGEFLATTENDATASET_H
