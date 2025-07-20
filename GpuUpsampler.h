// GpuUpsampler.h
#pragma once
#include <cstdint>

enum class ResampleKernel {
    Linear,
    Cubic,
    Sinc
};

class GpuUpsampler {
public:
    virtual bool initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) = 0;
    virtual void setKernel(ResampleKernel kernel) = 0;

    // outputFrames should be preallocated and updated with final count
    virtual bool process(const float* input, uint32_t inputFrames,
        float* output, uint32_t& outputFrames) = 0;

    virtual void shutdown() = 0;
    virtual ~GpuUpsampler() = default;
};
