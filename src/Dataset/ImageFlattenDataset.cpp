//
// Created by vlad on 4/25/23.
//

#include <filesystem>
#include <iostream>
#include <set>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include "Dataset/ImageFlattenDataset.h"

ImageFlattenDataset::ImageFlattenDataset(std::string folderPath, int imageHeight, int imageWidth, int batchSize,
                                         int fixedSize, int seed)
    : Dataset(folderPath, batchSize, seed), imageHeight(imageHeight), imageWidth(imageWidth)  {
    std::set<std::string> uniqueLabels;
    for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(folderPath)){
        if(is_directory(dirEntry.path())){
            continue;
        }
        std::string labelName = dirEntry.path().parent_path().filename();
        if(uniqueLabels.find(labelName) == uniqueLabels.end()){
            uniqueLabels.insert(labelName);
            labelsNames.push_back(labelName);
        }
        imagePaths.push_back(dirEntry.path());
    }
    std::sort(labelsNames.begin(), labelsNames.end());
    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(seed);
    std::shuffle(imagePaths.begin(), imagePaths.end(), g);
    if(fixedSize > 0){
        imagePaths.erase(imagePaths.begin() + fixedSize + 1, imagePaths.end());
    }

    std::vector<Matrix::Ptr> singleBatchData;
    std::vector<Matrix::Ptr> singleBatchLabel;
    for (const auto& imagePath : imagePaths) {
        try {
            singleBatchData.push_back(preprocessImage(imagePath));
            singleBatchLabel.push_back(preprocessLabel(imagePath));
        }
        catch (std::exception& e){

        }
    }

    if(batchSize == -1){
        this->batchSize = singleBatchData.size();
    }
    int datasetSize = singleBatchData.size();
    int batchNum = (datasetSize + this->batchSize - 1) / this->batchSize;
    for (int i = 0; i < batchNum; ++i) {
        auto dataBegin = singleBatchData.begin() + i * this->batchSize;
        auto dataEnd = singleBatchData.begin() + std::min((i + 1) * this->batchSize, datasetSize);
        data.push_back(Matrix::merge(dataBegin, dataEnd, 0));

        auto labelBegin = singleBatchLabel.begin() + i * this->batchSize;
        auto labelEnd = singleBatchLabel.begin() +
                                                     std::min((i + 1) * this->batchSize, datasetSize);
        labels.push_back(Matrix::merge(labelBegin, labelEnd, 0));
    }
}

Matrix::Ptr ImageFlattenDataset::preprocessImage(std::string imagePath) {
    cv::Mat image = imread(imagePath, cv::IMREAD_GRAYSCALE );
    cv::resize(image, image, cv::Size(imageHeight, imageWidth));
    image.convertTo(image, CV_32F, 1.0 / 255, 0);
    float* data = image.ptr<float>(0);
    Matrix::Ptr matrix(new Matrix(data, imageHeight * imageWidth, 1));
    return matrix;
}

Matrix::Ptr ImageFlattenDataset::preprocessLabel(std::string imagePath) {
    Matrix::Ptr label(new Matrix(1, 1));
    std::string labelName = std::filesystem::path(imagePath).parent_path().filename();
    auto it = std::find(labelsNames.begin(), labelsNames.end(), labelName);;
    label->get(0, 0) = std::distance(labelsNames.begin(), it) ;
    return label;
}
