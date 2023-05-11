//
// Created by vlad on 5/6/23.
//

#ifndef TRAINNN_MONITORING_H
#define TRAINNN_MONITORING_H

#include <vector>
#include <string>
#include <chrono>
#include "LogSample.h"

enum class Verbose {
    None,
    All
};

class Monitoring {
    Verbose logLevel;
    std::vector<LogSample> history;
    int batchSize{};
    int batchCount{};
    int params{};
    std::chrono::high_resolution_clock::time_point firstTimeStep;
    std::chrono::high_resolution_clock::time_point lastTimeStep;

public:
    explicit Monitoring(int batchSize, int batchCount, int params, Verbose logLevel = Verbose::All);

    void add(int epoch, int batch_no,
             float loss = std::numeric_limits<float>::lowest(),
             float val_loss = std::numeric_limits<float>::lowest(),
             float val_acc = std::numeric_limits<float>::lowest());

    void logLastSample();

    void serialize(const std::string& logDir);
};


#endif //TRAINNN_MONITORING_H
