//
// Created by vlad on 4/25/23.
//

#include "Dataset/Dataset.h"

Dataset::Dataset(std::string folderPath, int batchSize, int seed)
    : folderPath(folderPath), batchSize(batchSize), seed(seed) {

}

std::vector<Matrix::Ptr> Dataset::getData() {
    return data;
}

std::vector<Matrix::Ptr> Dataset::getLabel() {
    return labels;
}
