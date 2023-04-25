//
// Created by vlad on 4/25/23.
//

#ifndef TRAINNN_DATASET_H
#define TRAINNN_DATASET_H

#include <string>
#include <vector>
#include "Matrix/Matrix.h"

enum class DatasetType{
    Train,
    Inference
};

class Dataset {
protected:
    int batchSize = -1;
    int seed;
    std::string folderPath;
    std::vector<Matrix::Ptr> data;
    std::vector<Matrix::Ptr> labels;
    std::vector<std::string> labelsNames;
    DatasetType datasetType;
public:
    typedef std::unique_ptr<Dataset> Ptr;

    Dataset(std::string folderPath, int batchSize, int seed);
    std::vector<Matrix::Ptr> getData();
    std::vector<Matrix::Ptr> getLabel();
    virtual Matrix::Ptr preprocessImage(std::string imagePath) = 0;
    virtual Matrix::Ptr preprocessLabel(std::string imagePath) = 0;
    virtual ~Dataset() = default;
};


#endif //TRAINNN_DATASET_H
