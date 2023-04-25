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
    ImageFlattenDataset(std::string folderPath, int imageHeight, int imageWidth, int batchSize = -1, int seed = 42);
    std::vector<Matrix::Ptr> getData();
    std::vector<Matrix::Ptr> getLabel();
    Matrix::Ptr preprocessImage(std::string imagePath);
    Matrix::Ptr preprocessLabel(std::string imagePath);
};


#endif //TRAINNN_IMAGEFLATTENDATASET_H
