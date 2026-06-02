#pragma once

#include "../Audio/FloatRingBuffer.h"
#include "../Audio/OutputBufferQueue.h"
#include "../VulkanUpsampler.h"

#include <atomic>
#include <cstdint>
#include <thread>

struct GpuProcessingContext
{
    std::atomic<bool> *gpuReady = nullptr;
    VulkanUpsampler *upsampler = nullptr;
    FloatRingBuffer *inputRing = nullptr;
    OutputBufferQueue *outputQueue = nullptr;
    std::atomic<uint64_t> *processedInputFrames = nullptr;
    std::atomic<uint64_t> *processedOutputFrames = nullptr;
    uint32_t inputSampleRate = 0;
    uint32_t outputSampleRate = 0;
    uint32_t channels = 0;
    uint32_t outputRingCapacityFrames = 0;
    uint32_t idleSleepMs = 0;
};

class GpuProcessingThread
{
  public:
    explicit GpuProcessingThread(GpuProcessingContext *context);
    ~GpuProcessingThread();

    void start();
    void stop();

  private:
    void runLoop();

    GpuProcessingContext *context = nullptr;
    std::atomic<bool> running{false};
    std::thread thread;
};
