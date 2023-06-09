//
// Created by vlad on 4/25/23.
//

#include "Dataset/ImageFlattenDataset.h"

#include <filesystem>
#include <iostream>
#include <set>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>

ImageFlattenDataset::ImageFlattenDataset(const std::string& folderPath,
                                         int imageHeight, int imageWidth, int batchSize, int seed)
        : Dataset(folderPath, batchSize, seed), imageHeight(imageHeight), imageWidth(imageWidth) {
}

ImageFlattenDataset::Ptr
ImageFlattenDataset::createDataset(const std::string& folderPath, int imageHeight, int imageWidth, int batchSize,
                                   int fixedSize, int seed) {
    ImageFlattenDataset::Ptr dataset(new ImageFlattenDataset(folderPath, imageHeight, imageWidth, batchSize, seed));
    std::set<std::string> uniqueLabels;
    for (const auto& dirEntry: std::filesystem::recursive_directory_iterator(folderPath)) {
        if (is_directory(dirEntry.path())) {
            std::string labelName = dirEntry.path().filename();
            if (uniqueLabels.find(labelName) == uniqueLabels.end()) {
                uniqueLabels.insert(labelName);
                dataset->labelsNames.push_back(labelName);
            }
            continue;
        }
        dataset->imagePaths.push_back(dirEntry.path());
    }
    std::sort(dataset->labelsNames.begin(), dataset->labelsNames.end());
    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(seed);
    std::shuffle(dataset->imagePaths.begin(), dataset->imagePaths.end(), g);
    if (fixedSize > 0) {
        dataset->imagePaths.erase(dataset->imagePaths.begin() + fixedSize, dataset->imagePaths.end());
    }

    std::vector<Matrix::Ptr> singleBatchData;
    std::vector<Matrix::Ptr> singleBatchLabel;
    for (const auto& imagePath: dataset->imagePaths) {
        try {
            singleBatchData.push_back(dataset->preprocessImage(imagePath));
            singleBatchLabel.push_back(dataset->preprocessLabel(imagePath));
        }
        catch (std::exception& e) {

        }
    }

    if (batchSize == -1) {
        dataset->batchSize = static_cast<int>(singleBatchData.size());
    }
    int datasetSize = static_cast<int>(singleBatchData.size());
    int batchNum = (datasetSize + dataset->batchSize - 1) / dataset->batchSize;
    for (int i = 0; i < batchNum; ++i) {
        auto dataBegin = singleBatchData.begin() + i * dataset->batchSize;
        auto dataEnd = singleBatchData.begin() + std::min((i + 1) * dataset->batchSize, datasetSize);
        dataset->data.push_back(Matrix::merge(dataBegin, dataEnd, 0));

        auto labelBegin = singleBatchLabel.begin() + i * dataset->batchSize;
        auto labelEnd = singleBatchLabel.begin() +
                        std::min((i + 1) * dataset->batchSize, datasetSize);
        dataset->labels.push_back(Matrix::merge(labelBegin, labelEnd, 0));
    }
    return dataset;
}

Matrix ImageFlattenDataset::preprocessImage(const std::string& imagePath, int height, int width) {
    cv::Mat image = imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(image, CV_32F, 1.0 / 255, 0);
    auto* data = image.ptr<float>(0);
    Matrix matrix(data, height * width, 1);
    return matrix;
}

Matrix::Ptr ImageFlattenDataset::preprocessImage(std::string imagePath) {
    cv::Mat image = imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(imageHeight, imageWidth));
    image.convertTo(image, CV_32F, 1.0 / 255, 0);
    auto* data = image.ptr<float>(0);
    Matrix::Ptr matrix(new Matrix(data, imageHeight * imageWidth, 1));
    return matrix;
}

Matrix::Ptr ImageFlattenDataset::preprocessLabel(std::string imagePath) {
    Matrix::Ptr label(new Matrix(1, 1));
    if (labelsNames.size() > 2) {
        label.reset(new Matrix(static_cast<int>(labelsNames.size()), 1));
    }
    label->zeroInit();
    std::string labelName = std::filesystem::path(imagePath).parent_path().filename();
    auto it = std::find(labelsNames.begin(), labelsNames.end(), labelName);
    if (labelsNames.size() == 2) {
        label->get(0, 0) = static_cast<float>(std::distance(labelsNames.begin(), it));
    } else {
        int classIndex = static_cast<int>(std::distance(labelsNames.begin(), it));
        label->get(classIndex, 0) = 1;
    }
    return label;
}
