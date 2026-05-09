#pragma once

#include "../Audio/AudioCallbackContext.h"
#include "../Audio/AudioDeviceManager.h"
#include "../Audio/FloatRingBuffer.h"
#include "../GpuUpsampler.h"
#include "../RenderPipeline/GpuProcessingThread.h"

#include <atomic>
#include <chrono>
#include <cstdint>

#include <memory>

class AppRuntime
{
  public:
    AppRuntime();

    bool Initialize();
    int Run();
    void Shutdown();
    void RequestStop();

  private:
    void resetRecoveryTelemetry(uint32_t initialMinOutputFrames);
    void recalibrateAdaptiveTarget(const char *reason);
    bool recoverPlaybackAfterUnderrun();
    void waitForOutputPrebuffer() const;
    void runMainLoop();

    std::atomic<bool> running{true};
    std::atomic<bool> underrun{false};
    std::atomic<bool> restartRequested{false};
    std::atomic<bool> restartInProgress{false};
    std::atomic<bool> gpuReady{false};
    std::atomic<uint64_t> capturedInputFrames{0};
    std::atomic<uint64_t> processedInputFrames{0};
    std::atomic<uint64_t> processedOutputFrames{0};
    std::atomic<uint64_t> playedOutputFrames{0};
    std::atomic<uint64_t> zeroFillEvents{0};
    std::atomic<uint64_t> zeroFillSamples{0};
    std::atomic<uint32_t> minObservedOutputFrames{UINT32_MAX};

    std::unique_ptr<GpuUpsampler> upsampler;
    FloatRingBuffer inputRing;
    FloatRingBuffer outputRing;
    GpuProcessingContext gpuProcessingContext{};
    GpuProcessingThread gpuProcessor;

    AudioCallbackContext audioCallbackContext{};
    std::unique_ptr<AudioDeviceManager> deviceManager;
    bool initialized = false;
};
