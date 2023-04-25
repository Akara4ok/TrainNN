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
                                         int seed)
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

    std::vector<Matrix::Ptr> singleBatchData;
    std::vector<Matrix::Ptr> singleBatchLabel;
    for (const auto& imagePath : imagePaths) {
        singleBatchData.push_back(preprocessImage(imagePath));
        singleBatchLabel.push_back(preprocessLabel(imagePath));
    }

    data.push_back(Matrix::merge(singleBatchData.begin(), singleBatchData.end(), 0));
    labels.push_back(Matrix::merge(singleBatchLabel.begin(), singleBatchLabel.end(), 0));
}

std::vector<Matrix::Ptr> ImageFlattenDataset::getData() {
    return Dataset::getData();
}

std::vector<Matrix::Ptr> ImageFlattenDataset::getLabel() {
    return Dataset::getLabel();
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
