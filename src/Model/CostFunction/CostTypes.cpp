//
// Created by vlad on 4/25/23.
//

#include "Model/CostFunction/CostTypes.h"

std::ostream& operator<<(std::ostream& os, const Cost& other)
{
    switch (other)
    {
        case Cost::BinaryCrossEntropy:
            os << "BinaryCrossEntropy";
            break;
        default:
            break;
    }
    return os;
};

std::istream& operator>> (std::istream& is, Cost& type)
{
    std::string input;
    is >> input;
    if(input == "BinaryCrossEntropy") {
        type = Cost::BinaryCrossEntropy;
    }
    return is;
};
