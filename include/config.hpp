//
// Created by vlad on 4/23/23.
//

#ifndef CMAKE_AND_CUDA_CONFIG_H
#define CMAKE_AND_CUDA_CONFIG_H

enum class Provider {
    CPU,
    GPU
};

class Config {
    Provider provider = Provider::CPU;

    Config() {};

public:
    Config(Config& other) = delete;

    void operator=(const Config&) = delete;

    static Config& getInstance() {
        static Config c_Instance;
        return c_Instance;
    }

    Provider getProvider() {
        return provider;
    };

    void setProvider(Provider prov) {
        provider = prov;
    }
};

#endif //CMAKE_AND_CUDA_CONFIG_H
