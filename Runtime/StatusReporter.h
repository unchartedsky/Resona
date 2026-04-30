#pragma once

#include <cstdint>

struct RuntimeStatusSnapshot
{
    uint32_t outputBufferFrames = 0;
    double outputFillRate = 0.0;
    uint32_t busySlots = 0;
    uint32_t totalSlots = 0;
    float currentRatio = 0.0f;
    float baseRatio = 0.0f;
    float targetFillRatio = 0.0f;
    uint32_t outputRingCapacityFrames = 0;
    uint32_t minObservedOutputFrames = 0;
    uint64_t zeroFillEvents = 0;
    uint64_t zeroFillSamples = 0;
    long long totalElapsedSeconds = 0;
};

struct SessionStatistics
{
    long long totalElapsedSeconds = 0;
    uint64_t totalCapturedFrames = 0;
    uint64_t totalProcessedFrames = 0;
    uint64_t totalPlayedFrames = 0;
    uint32_t minObservedOutputFrames = 0;
    uint64_t zeroFillEvents = 0;
    uint64_t zeroFillSamples = 0;
};

class StatusReporter
{
  public:
    static void PrintStatusLine(const RuntimeStatusSnapshot &snapshot);
    static void PrintSessionStatistics(const SessionStatistics &statistics);
};
