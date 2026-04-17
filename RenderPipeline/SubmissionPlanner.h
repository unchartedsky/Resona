#pragma once

#include <cstdint>

struct SubmissionInputs
{
    uint32_t outputBufferFrames = 0;
    uint32_t outputRingCapacityFrames = 0;
    uint32_t availableInputFrames = 0;
    uint32_t availableSlots = 0;
    uint32_t batchFrames = 0;
    uint32_t expectedOutputFramesPerBatch = 0;
    uint32_t idleSleepMs = 0;
    uint32_t targetBufferPercent = 0;
    uint32_t maxSubmittedBatches = 0;
    float submitCriticalThreshold = 0.0f;
    float submitLowThreshold = 0.0f;
    float sleepCriticalThreshold = 0.0f;
    float sleepLowThreshold = 0.0f;
};

struct SubmissionPlan
{
    uint32_t targetBufferFrames = 0;
    float bufferRatio = 0.0f;
    uint32_t batchesToSubmit = 0;
    uint32_t sleepMs = 0;
    bool shouldSubmit = false;
};

class SubmissionPlanner
{
  public:
    static SubmissionPlan Build(const SubmissionInputs &inputs);
};
