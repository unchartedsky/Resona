#pragma once
#include "GpuUpsampler.h"

class PassthroughUpsampler : public GpuUpsampler {
public:
    bool initialize(uint32_t inRate, uint32_t outRate, uint32_t ch) override {
        inputRate = inRate;
        outputRate = outRate;
        channels = ch;
        return true;
    }

    void setKernel(ResampleKernel) override {}

    void shutdown() override {}

private:
    uint32_t inputRate = 0;
    uint32_t outputRate = 0;
    uint32_t channels = 0;
};
