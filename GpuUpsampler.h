#pragma once
#include <cstdint>

/// @brief Available resampling kernel types.
enum class ResampleKernel {
    Linear,
    Cubic,
    Sinc
};

/// @brief Abstract interface for GPU-based audio resamplers.
class GpuUpsampler {
public:
    virtual ~GpuUpsampler() = default;

    /// @brief Initializes the upsampler with given I/O settings.
    /// @param inputRate   Input sample rate in Hz.
    /// @param outputRate  Output sample rate in Hz.
    /// @param channels    Number of audio channels (e.g., 2 for stereo).
    virtual bool initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) = 0;

    /// @brief Sets the active resampling kernel.
    /// @param kernel  The resample algorithm to use.
    virtual void setKernel(ResampleKernel kernel) = 0;

    /// @brief Processes input audio and writes upsampled output.
    /// @param input          Pointer to interleaved input samples (float32).
    /// @param inputFrames    Number of input frames (not samples).
    /// @param output         Pointer to preallocated output buffer (float32).
    /// @param outputFrames   Updated with number of output frames produced.
    /// @return true if successful, false otherwise.
    virtual bool process(const float* input, uint32_t inputFrames,
        float* output, uint32_t& outputFrames) = 0;

    /// @brief Releases GPU resources and shuts down the upsampler.
    virtual void shutdown() = 0;
};
