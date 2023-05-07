//
// Created by vlad on 5/6/23.
//

#ifndef TRAINNN_LOGSAMPLE_H
#define TRAINNN_LOGSAMPLE_H

#include <limits>
#include <string>
#include <iostream>

struct LogSample {
    int epoch;
    int batch_no;
    float loss;
    float val_loss;
    float val_acc;
    float duration;

    LogSample(int epoch, int batch_no, float duration,
              float loss = std::numeric_limits<float>::lowest(),
              float val_loss = std::numeric_limits<float>::lowest(),
              float val_acc = std::numeric_limits<float>::lowest())
            : epoch(epoch), batch_no(batch_no), duration(duration), loss(loss), val_loss(val_loss), val_acc(val_acc) {
    }

    static std::string getSerializableNames();

    friend std::ostream& operator<<(std::ostream& os, const LogSample& log);
};


#endif //TRAINNN_LOGSAMPLE_H
