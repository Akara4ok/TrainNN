//
// Created by vlad on 5/6/23.
//

#include "Model/Monitoring/LogSample.h"

std::string LogSample::getSerializableNames() {
    return "epoch,batchNo,loss,val_loss,val_acc,duration\n";
}

std::ostream& operator<<(std::ostream& os, const LogSample& log) {
    os << log.epoch << ",";
    if (log.batchNo == -1) {
        os << ",," << log.val_loss << "," << log.val_acc << ",";
    } else {
        os << log.batchNo << "," << log.loss << ",,,";
    }
    os << log.duration << "\n";
    return os;
}
