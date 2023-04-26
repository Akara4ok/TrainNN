//
// Created by vlad on 4/25/23.
//

#include "Dataset/Dataset.h"

#include <utility>

Dataset::Dataset(std::string folderPath, int batchSize, int seed)
        : folderPath(std::move(folderPath)), batchSize(batchSize), seed(seed) {

}

std::vector<Matrix::Ptr> Dataset::getData() {
    return data;
}

std::vector<Matrix::Ptr> Dataset::getLabel() {
    return labels;
}
