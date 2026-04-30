#pragma once

#include "FloatRingBuffer.h"

#include <atomic>
#include <cstdint>

struct AudioCallbackContext
{
    std::atomic<bool> *running = nullptr;
    std::atomic<bool> *underrun = nullptr;

    FloatRingBuffer *inputRing = nullptr;
    FloatRingBuffer *outputRing = nullptr;

    std::atomic<uint64_t> *capturedInputFrames = nullptr;
    std::atomic<uint64_t> *playedOutputFrames = nullptr;
    std::atomic<uint64_t> *zeroFillEvents = nullptr;
    std::atomic<uint64_t> *zeroFillSamples = nullptr;
    std::atomic<uint32_t> *minObservedOutputFrames = nullptr;

    uint32_t channels = 0;
    uint32_t minBufferFrames = 0;
};
