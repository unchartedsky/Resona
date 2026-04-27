#include "GpuProcessingThread.h"

#include "SubmissionPlanner.h"

#include <chrono>
#include <cstdio>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

GpuProcessingThread::GpuProcessingThread(GpuProcessingContext *context)
    : context(context)
{
}

GpuProcessingThread::~GpuProcessingThread()
{
    stop();
}

void GpuProcessingThread::start()
{
    if (running)
        return;

    running = true;
    thread = std::thread([this]() { runLoop(); });
}

void GpuProcessingThread::stop()
{
    if (!running)
        return;

    running = false;
    if (thread.joinable())
    {
        thread.join();
    }
}

void GpuProcessingThread::runLoop()
{
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif

    printf("[+] GPU processing thread started (adaptive mode)\n");

    std::vector<float> inputBuffer;

    while (running)
    {
        if (!context || !context->gpuReady || !context->upsampler || !context->inputRing || !context->outputRing)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (!context->gpuReady->load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        const uint32_t outputBufferFrames = context->outputRing->available() / context->channels;
        const uint32_t batchFrames = VulkanUpsampler::AdaptivePolicy::FixedBatchFrames;
        const uint32_t batchSamples = batchFrames * context->channels;
        const uint32_t availableInputFrames = context->inputRing->available() / context->channels;
        const size_t availableSlots = context->upsampler->getAvailableSlots();
        const float ratio = static_cast<float>(context->outputSampleRate) / context->inputSampleRate;
        const uint32_t expectedOutputFramesPerBatch = static_cast<uint32_t>(batchFrames * ratio);

        SubmissionInputs plannerInputs{};
        plannerInputs.outputBufferFrames = outputBufferFrames;
        plannerInputs.outputRingCapacityFrames = context->outputRingCapacityFrames;
        plannerInputs.availableInputFrames = availableInputFrames;
        plannerInputs.availableSlots = static_cast<uint32_t>(availableSlots);
        plannerInputs.batchFrames = batchFrames;
        plannerInputs.expectedOutputFramesPerBatch = expectedOutputFramesPerBatch;
        plannerInputs.idleSleepMs = context->idleSleepMs;
        plannerInputs.targetBufferPercent = VulkanUpsampler::AdaptivePolicy::TargetBufferPercent;
        plannerInputs.maxSubmittedBatches = VulkanUpsampler::AdaptivePolicy::MaxSubmittedBatches;
        plannerInputs.submitCriticalThreshold = VulkanUpsampler::AdaptivePolicy::SubmitCriticalThreshold;
        plannerInputs.submitLowThreshold = VulkanUpsampler::AdaptivePolicy::SubmitLowThreshold;
        plannerInputs.sleepCriticalThreshold = VulkanUpsampler::AdaptivePolicy::SleepCriticalThreshold;
        plannerInputs.sleepLowThreshold = VulkanUpsampler::AdaptivePolicy::SleepLowThreshold;

        const SubmissionPlan plan = SubmissionPlanner::Build(plannerInputs);

        context->upsampler->updateAdaptiveParams(outputBufferFrames, plan.targetBufferFrames);

        if (!plan.shouldSubmit)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(context->idleSleepMs));
            continue;
        }

        const uint32_t maxBatchSamples = batchSamples * plan.batchesToSubmit;
        if (inputBuffer.size() < maxBatchSamples)
        {
            inputBuffer.resize(maxBatchSamples);
        }

        for (uint32_t i = 0; i < plan.batchesToSubmit; ++i)
        {
            const uint32_t readSamples = context->inputRing->pop(inputBuffer.data() + (i * batchSamples), batchSamples);
            const uint32_t readFrames = readSamples / context->channels;

            if (readFrames == 0)
            {
                break;
            }

            uint64_t sequenceId = context->upsampler->processAsync(
                inputBuffer.data() + (i * batchSamples), readFrames,
                [this, readFrames = readFrames](const float *output, uint32_t outputFrames) {
                    const uint32_t outputSamples = outputFrames * context->channels;
                    context->outputRing->push(output, outputSamples);

                    if (context->processedInputFrames)
                    {
                        context->processedInputFrames->fetch_add(readFrames, std::memory_order_relaxed);
                    }
                    if (context->processedOutputFrames)
                    {
                        context->processedOutputFrames->fetch_add(outputFrames, std::memory_order_relaxed);
                    }
                });

            if (sequenceId == 0)
            {
                break;
            }
        }

        context->upsampler->tryPollAll();

        if (plan.sleepMs > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(plan.sleepMs));
        }
    }

    printf("[+] GPU processing thread stopped\n");
}
