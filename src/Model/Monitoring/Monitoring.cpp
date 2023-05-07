//
// Created by vlad on 5/6/23.
//

#include "Model/Monitoring/Monitoring.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "config.hpp"

Monitoring::Monitoring(int batchSize, int batchCount, int params, Verbose logLevel)
: batchSize(batchSize), batchCount(batchCount), params(params), logLevel(logLevel) {
    firstTimeStep = std::chrono::high_resolution_clock::now();
    lastTimeStep = std::chrono::high_resolution_clock::now();
}

void Monitoring::add(int epoch, int batch_no, float loss, float val_loss, float val_acc) {
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_float = t2 - lastTimeStep;
    history.emplace_back(epoch, batch_no, duration_float.count(), loss, val_loss, val_acc);
    lastTimeStep = t2;
}

void Monitoring::logLastSample() {
    if(logLevel == Verbose::All){
        LogSample lastLog = history.back();
        if(lastLog.batch_no == 0){
            std::cout << "Epoch: " << lastLog.epoch << ": ";
            if(batchCount > 1){
                std::cout << "\n";
            }
        }
        if(lastLog.batch_no >= 0 &&  batchCount > 1){
            std::cout << "   Batch: " << lastLog.batch_no << ": ";
        }
        if(lastLog.batch_no >= 0){
            std::cout << "loss: " << lastLog.loss << " ";
        }
        if(lastLog.batch_no == -1){
            if(batchCount > 1){
                std::cout << "Epoch: " << lastLog.epoch << ": ";
            } else {
                std::cout << "-- ";
            }
            std::cout << "val_loss: " << lastLog.val_loss << " ";
            std::cout << "- val_accuracy: " << lastLog.val_acc << std::endl;
        }
        if(batchCount > 1){
            std::cout << std::endl;
        }
    }
}

void Monitoring::serialize(std::string logDir) {
    if (!std::filesystem::is_directory(logDir)) {
        std::filesystem::create_directories(logDir);
    }
    std::ofstream history_out(logDir + "/history.csv");
    history_out << LogSample::getSerializableNames();
    for (const auto& sample : history) {
        history_out << sample;
    }

    if (!std::filesystem::is_directory(Config::getInstance().getLogDir())) {
        std::filesystem::create_directories(logDir);
    }
    bool metadataExists = std::filesystem::exists(Config::getInstance().getLogDir() + "/metadata.csv");
    std::ofstream metadata_out(Config::getInstance().getLogDir() + "/metadata.csv", std::ios_base::app);
    if(!metadataExists){
        metadata_out << "log_dir,epochs,batch_size,batch_count,params,total_time\n";
    }
    metadata_out << logDir << "," << history.back().epoch + 1 << ","
        << batchSize << "," << batchCount << "," << params << ",";

    std::chrono::duration<float> duration_float = lastTimeStep - firstTimeStep;
    metadata_out << duration_float.count() << "\n";
}
