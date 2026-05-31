#include "SubmissionPlanner.h"

#include <algorithm>

namespace
{
struct SubmissionRule
{
    float maxBufferRatio;
    uint32_t minAvailableSlots;
    uint32_t targetBatches;
};

struct SleepRule
{
    float maxBufferRatio;
    uint32_t sleepMs;
};

uint32_t ComputeBatchesToSubmit(const SubmissionInputs &inputs, float bufferRatio)
{
    if (inputs.availableSlots == 0 || inputs.availableInputFrames < inputs.batchFrames)
    {
        return 0;
    }

    const SubmissionRule resolvedRules[] = {
        {inputs.submitCriticalThreshold, 4u, inputs.maxSubmittedBatches},
        {inputs.submitLowThreshold, 3u, 3u},
        {1.0f, 2u, 2u},
    };

    uint32_t batchesToSubmit = 1;
    for (const SubmissionRule &rule : resolvedRules)
    {
        if (bufferRatio < rule.maxBufferRatio && inputs.availableSlots >= rule.minAvailableSlots)
        {
            batchesToSubmit = std::min(inputs.availableSlots, rule.targetBatches);
            break;
        }
    }

    const uint32_t totalRequiredFrames = inputs.batchFrames * batchesToSubmit;
    if (inputs.availableInputFrames < totalRequiredFrames)
    {
        batchesToSubmit = std::max(1u, inputs.availableInputFrames / inputs.batchFrames);
    }

    return batchesToSubmit;
}

uint32_t ComputeSleepMs(const SubmissionInputs &inputs, float bufferRatio)
{
    const SleepRule rules[] = {
        {inputs.sleepCriticalThreshold, 0u},
        {inputs.sleepLowThreshold, 1u},
        {2.0f, inputs.idleSleepMs},
    };

    for (const SleepRule &rule : rules)
    {
        if (bufferRatio < rule.maxBufferRatio)
        {
            return rule.sleepMs;
        }
    }

    return inputs.idleSleepMs;
}
} // namespace

SubmissionPlan SubmissionPlanner::Build(const SubmissionInputs &inputs)
{
    SubmissionPlan plan{};

    if (inputs.outputRingCapacityFrames == 0 || inputs.batchFrames == 0)
    {
        plan.sleepMs = inputs.idleSleepMs;
        return plan;
    }

    plan.targetBufferFrames = inputs.outputRingCapacityFrames * inputs.targetBufferPercent / 100;

    if (plan.targetBufferFrames > 0)
    {
        plan.bufferRatio = static_cast<float>(inputs.outputBufferFrames) / plan.targetBufferFrames;
    }

    const uint32_t batchesToSubmit = ComputeBatchesToSubmit(inputs, plan.bufferRatio);

    if (batchesToSubmit == 0)
    {
        plan.batchesToSubmit = 0;
        plan.sleepMs = inputs.idleSleepMs;
        return plan;
    }

    const uint32_t outputFreeFrames = inputs.outputRingCapacityFrames - inputs.outputBufferFrames;
    const uint32_t totalExpectedOutput = inputs.expectedOutputFramesPerBatch * batchesToSubmit;
    if (outputFreeFrames < totalExpectedOutput)
    {
        plan.batchesToSubmit = 0;
        plan.sleepMs = inputs.idleSleepMs;
        return plan;
    }

    plan.batchesToSubmit = batchesToSubmit;
    plan.shouldSubmit = true;
    plan.sleepMs = ComputeSleepMs(inputs, plan.bufferRatio);

    return plan;
}
