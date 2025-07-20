// PassthroughUpsampler.h
#pragma once
#include "GpuUpsampler.h"
#include <cstring>

class PassthroughUpsampler : public GpuUpsampler {
public:
    bool initialize(uint32_t inRate, uint32_t outRate, uint32_t ch) override {
        inputRate = inRate;
        outputRate = outRate;
        channels = ch;
        return true;
    }

    void setKernel(ResampleKernel) override {}

    bool process(const float* input, uint32_t inputFrames,
        float* output, uint32_t& outputFrames) override {
        uint32_t frameCount = inputFrames;
        std::memcpy(output, input, frameCount * channels * sizeof(float));
        outputFrames = frameCount;
        return true;
    }

    void shutdown() override {}

private:
    uint32_t inputRate = 0;
    uint32_t outputRate = 0;
    uint32_t channels = 0;
};
