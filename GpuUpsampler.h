#pragma once
#include <cstdint>

/// @brief Available resampling kernel types.
enum class ResampleKernel {
    Nearest,  // Nearest neighbor (no interpolation, noise-free)
    Linear,   // Linear interpolation (smooth but may have artifacts)
    Cubic,    // Cubic interpolation (not yet implemented)
    Sinc      // Sinc interpolation (not yet implemented)
};

struct GpuUpsamplerRuntimeStatus {
    uint32_t busySlots = 0;
    uint32_t totalSlots = 0;
    float currentRatio = 0.0f;
    uint32_t targetBufferLevel = 0;
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

    /// @brief Return runtime status used by orchestration/status reporting.
    /// Default implementation reports an empty status for backends that do not expose these metrics.
    virtual GpuUpsamplerRuntimeStatus getRuntimeStatus() const { return {}; }

    /// @brief Releases GPU resources and shuts down the upsampler.
    virtual void shutdown() = 0;
};
