#include "SubmissionPlanner.h"

#include <algorithm>

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

    uint32_t batchesToSubmit = 1;
    if (plan.bufferRatio < inputs.submitCriticalThreshold && inputs.availableSlots > 3)
    {
        batchesToSubmit = std::min(inputs.availableSlots, inputs.maxSubmittedBatches);
    }
    else if (plan.bufferRatio < inputs.submitLowThreshold && inputs.availableSlots > 2)
    {
        batchesToSubmit = std::min(inputs.availableSlots, 3u);
    }
    else if (inputs.availableSlots > 1)
    {
        batchesToSubmit = std::min(inputs.availableSlots, 2u);
    }

    const uint32_t totalRequiredFrames = inputs.batchFrames * batchesToSubmit;
    if (inputs.availableInputFrames < totalRequiredFrames)
    {
        batchesToSubmit = std::max(1u, inputs.availableInputFrames / inputs.batchFrames);
    }

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

    if (plan.bufferRatio < inputs.sleepCriticalThreshold)
    {
        plan.sleepMs = 0;
    }
    else if (plan.bufferRatio < inputs.sleepLowThreshold)
    {
        plan.sleepMs = 1;
    }
    else
    {
        plan.sleepMs = inputs.idleSleepMs;
    }

    return plan;
}
