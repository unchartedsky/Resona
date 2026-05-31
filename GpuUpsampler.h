#pragma once
#include <cstdint>

/// @brief Available resampling kernel types.
enum class ResampleKernel {
    Nearest,  // Nearest neighbor (no interpolation, noise-free)
    Linear,   // Linear interpolation (smooth but may have artifacts)
    Cubic,    // Cubic interpolation (not yet implemented)
    Sinc      // Sinc interpolation (not yet implemented)
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

    /// @brief Reset any adaptive target/baseline capture used for runtime recalibration.
    /// Default implementation is a no-op for backends without adaptive target state.
    virtual void resetAdaptiveTarget() {}

    /// @brief Releases GPU resources and shuts down the upsampler.
    virtual void shutdown() = 0;
};
