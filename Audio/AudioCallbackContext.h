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

    uint32_t channels = 0;
    uint32_t minBufferFrames = 0;
};
