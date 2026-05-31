#include "AppRuntime.h"

#include "../Audio/AudioConfig.h"
#include "StatusReporter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

AppRuntime::AppRuntime()
    : gpuProcessor(&gpuProcessingContext)
{
}

void AppRuntime::resetRecoveryTelemetry(uint32_t initialMinOutputFrames)
{
    underrun.store(false, std::memory_order_relaxed);
    restartRequested.store(false, std::memory_order_relaxed);
    restartInProgress.store(false, std::memory_order_relaxed);
    zeroFillEvents.store(0, std::memory_order_relaxed);
    zeroFillSamples.store(0, std::memory_order_relaxed);
    minObservedOutputFrames.store(initialMinOutputFrames, std::memory_order_relaxed);
}

void AppRuntime::recalibrateAdaptiveTarget(const char *reason)
{
    if (!upsampler)
    {
        return;
    }

    if (reason && reason[0] != '\0')
    {
        printf("[*] Recalibrating adaptive target after %s\n", reason);
    }

    upsampler->resetAdaptiveTarget();
}

bool AppRuntime::recoverPlaybackAfterUnderrun()
{
    if (!deviceManager)
    {
        return false;
    }

    if (!deviceManager->stopPlayback())
    {
        return false;
    }

    waitForOutputPrebuffer();

    const uint32_t currentOutputFrames = outputRing.available() / AudioConfig::CHANNELS;
    resetRecoveryTelemetry(currentOutputFrames);
    recalibrateAdaptiveTarget("playback restart");
    return deviceManager->startPlayback();
}

void AppRuntime::waitForOutputPrebuffer() const
{
    const uint32_t targetFrames = AudioConfig::PREBUFFER_FRAMES;
    const uint32_t safetyMargin = targetFrames / 20;
    const uint32_t safeTargetFrames = targetFrames + safetyMargin;

    printf("[*] Waiting for prebuffer (target: %u frames + %u margin)...\n", targetFrames, safetyMargin);

    while (outputRing.available() / AudioConfig::CHANNELS < safeTargetFrames)
    {
        const uint32_t currentFrames = outputRing.available() / AudioConfig::CHANNELS;
        printf("    Progress: %u/%u frames (%.1f%%)     \r", currentFrames, safeTargetFrames,
               (currentFrames * 100.0f) / safeTargetFrames);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::SLEEP_INTERVAL_MS));
    }

    printf("\n[+] Prebuffer complete (filled: %u frames)                    \n",
           outputRing.available() / AudioConfig::CHANNELS);
}

void AppRuntime::runMainLoop()
{
    auto lastStatusTime = std::chrono::steady_clock::now();
    auto startTime = std::chrono::steady_clock::now();
    uint32_t lastOutputBufferLevel = 0;

    while (running)
    {
        if (underrun.exchange(false))
        {
            printf("\r%*s\r", 120, "");
            printf("[!] Underrun detected. Re-buffering...\n");

            restartRequested.store(true, std::memory_order_release);
        }

        if (restartRequested.load(std::memory_order_acquire) &&
            !restartInProgress.exchange(true, std::memory_order_acq_rel))
        {
            restartRequested.store(false, std::memory_order_release);

            if (recoverPlaybackAfterUnderrun())
            {
                printf("[+] Playback resumed after underrun\n");
                restartInProgress.store(false, std::memory_order_release);
            }
            else
            {
                printf("[!] Failed to restart playback after underrun\n");
                running.store(false, std::memory_order_release);
                restartInProgress.store(false, std::memory_order_release);
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatusTime).count();

        if (elapsedMs >= 500)
        {
            const uint32_t outputBufferFrames = outputRing.available() / AudioConfig::CHANNELS;
            const GpuUpsamplerRuntimeStatus upsamplerStatus = upsampler ? upsampler->getRuntimeStatus() : GpuUpsamplerRuntimeStatus{};

            const int32_t outputBufferDelta =
                static_cast<int32_t>(outputBufferFrames) - static_cast<int32_t>(lastOutputBufferLevel);
            const double outputFillRate = (outputBufferDelta * 1000.0) / elapsedMs;

            const uint32_t targetBufferFrames = upsamplerStatus.targetBufferLevel;
            float bufferPressure = 1.0f;

            if (targetBufferFrames > 0)
            {
                bufferPressure = std::min(1.0f, static_cast<float>(outputBufferFrames) / targetBufferFrames);
            }

            const float baseRatio = static_cast<float>(AudioConfig::OUTPUT_SAMPLE_RATE) /
                                    static_cast<float>(AudioConfig::INPUT_SAMPLE_RATE);

            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

            RuntimeStatusSnapshot statusSnapshot{};
            statusSnapshot.outputBufferFrames = outputBufferFrames;
            statusSnapshot.outputFillRate = outputFillRate;
            statusSnapshot.busySlots = upsamplerStatus.busySlots;
            statusSnapshot.totalSlots = upsamplerStatus.totalSlots;
            statusSnapshot.currentRatio = upsamplerStatus.currentRatio > 0.0f ? upsamplerStatus.currentRatio : baseRatio;
            statusSnapshot.baseRatio = baseRatio;
            statusSnapshot.targetFillRatio = bufferPressure;
            statusSnapshot.outputRingCapacityFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES;
            const uint32_t minOutputFrames = this->minObservedOutputFrames.load(std::memory_order_relaxed);
            statusSnapshot.minObservedOutputFrames =
                minOutputFrames == UINT32_MAX ? outputBufferFrames : minOutputFrames;
            statusSnapshot.zeroFillEvents = zeroFillEvents.load(std::memory_order_relaxed);
            statusSnapshot.zeroFillSamples = zeroFillSamples.load(std::memory_order_relaxed);
            statusSnapshot.totalElapsedSeconds = totalElapsed;
            StatusReporter::PrintStatusLine(statusSnapshot);

            lastOutputBufferLevel = outputBufferFrames;
            lastStatusTime = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::MAIN_LOOP_SLEEP_MS));
    }
    printf("\n");

    auto totalElapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count();
    const uint64_t totalCaptured = capturedInputFrames.load(std::memory_order_relaxed);
    const uint64_t totalProcessed = processedInputFrames.load(std::memory_order_relaxed);
    const uint64_t totalPlayed = playedOutputFrames.load(std::memory_order_relaxed);
    const uint32_t minOutputFrames = this->minObservedOutputFrames.load(std::memory_order_relaxed);

    SessionStatistics sessionStatistics{};
    sessionStatistics.totalElapsedSeconds = totalElapsed;
    sessionStatistics.totalCapturedFrames = totalCaptured;
    sessionStatistics.totalProcessedFrames = totalProcessed;
    sessionStatistics.totalPlayedFrames = totalPlayed;
    sessionStatistics.minObservedOutputFrames =
        minOutputFrames == UINT32_MAX ? 0 : minOutputFrames;
    sessionStatistics.zeroFillEvents = zeroFillEvents.load(std::memory_order_relaxed);
    sessionStatistics.zeroFillSamples = zeroFillSamples.load(std::memory_order_relaxed);
    StatusReporter::PrintSessionStatistics(sessionStatistics);
}

bool AppRuntime::Initialize()
{
    printf("[*] Initializing GPU upsampler...\n");

    upsampler = std::make_unique<VulkanUpsampler>();
    if (!upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE, AudioConfig::CHANNELS))
    {
        printf("[!] Failed to initialize upsampler\n");
        return false;
    }

    upsampler->setKernel(ResampleKernel::Nearest);

    printf("[*] Initializing dual ring buffers...\n");
    inputRing.init(AudioConfig::INPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    outputRing.init(AudioConfig::OUTPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    resetRecoveryTelemetry(AudioConfig::OUTPUT_RING_BUFFER_FRAMES);

    gpuProcessingContext.gpuReady = &gpuReady;
    gpuProcessingContext.upsampler = static_cast<VulkanUpsampler *>(upsampler.get());
    gpuProcessingContext.inputRing = &inputRing;
    gpuProcessingContext.outputRing = &outputRing;
    gpuProcessingContext.processedInputFrames = &processedInputFrames;
    gpuProcessingContext.processedOutputFrames = &processedOutputFrames;
    gpuProcessingContext.inputSampleRate = AudioConfig::INPUT_SAMPLE_RATE;
    gpuProcessingContext.outputSampleRate = AudioConfig::OUTPUT_SAMPLE_RATE;
    gpuProcessingContext.channels = AudioConfig::CHANNELS;
    gpuProcessingContext.outputRingCapacityFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES;
    gpuProcessingContext.idleSleepMs = AudioConfig::GPU_THREAD_SLEEP_MS;

    gpuReady.store(true, std::memory_order_release);

    printf("[*] Starting GPU processing thread...\n");
    gpuProcessor.start();

    printf("[*] Initializing audio devices...\n");
    audioCallbackContext.running = &running;
    audioCallbackContext.underrun = &underrun;
    audioCallbackContext.inputRing = &inputRing;
    audioCallbackContext.outputRing = &outputRing;
    audioCallbackContext.capturedInputFrames = &capturedInputFrames;
    audioCallbackContext.playedOutputFrames = &playedOutputFrames;
    audioCallbackContext.zeroFillEvents = &zeroFillEvents;
    audioCallbackContext.zeroFillSamples = &zeroFillSamples;
    audioCallbackContext.minObservedOutputFrames = &minObservedOutputFrames;
    audioCallbackContext.channels = AudioConfig::CHANNELS;
    audioCallbackContext.minBufferFrames = AudioConfig::MIN_BUFFER_FRAMES;

    deviceManager = std::make_unique<AudioDeviceManager>(&audioCallbackContext);
    if (!deviceManager->initializeCapture() || !deviceManager->initializePlayback())
    {
        gpuReady.store(false, std::memory_order_release);
        gpuProcessor.stop();
        deviceManager.reset();
        upsampler.reset();
        return false;
    }

    if (!deviceManager->startCapture())
    {
        gpuReady.store(false, std::memory_order_release);
        gpuProcessor.stop();
        deviceManager.reset();
        upsampler.reset();
        return false;
    }

    waitForOutputPrebuffer();
    resetRecoveryTelemetry(outputRing.available() / AudioConfig::CHANNELS);

    if (!deviceManager->startPlayback())
    {
        gpuReady.store(false, std::memory_order_release);
        gpuProcessor.stop();
        deviceManager->stopDevices();
        deviceManager.reset();
        upsampler.reset();
        return false;
    }

    recalibrateAdaptiveTarget("startup prebuffer");

    printf("[+] Real-time GPU upsampler started (%u -> %uHz)\n", AudioConfig::INPUT_SAMPLE_RATE,
           AudioConfig::OUTPUT_SAMPLE_RATE);
    printf("[*] Architecture: Capture -> Input Buffer (44.1kHz) -> GPU -> Output Buffer (384kHz) -> Playback\n");
    printf("[*] Press Ctrl+C to stop...\n");

    initialized = true;
    return true;
}

int AppRuntime::Run()
{
    if (!initialized || !deviceManager)
    {
        return EXIT_FAILURE;
    }

    runMainLoop();
    return EXIT_SUCCESS;
}

void AppRuntime::RequestStop()
{
    running.store(false, std::memory_order_release);
}

void AppRuntime::Shutdown()
{
    if (!initialized)
    {
        return;
    }

    printf("[*] Initiating shutdown sequence...\n");

    running.store(false, std::memory_order_release);
    gpuReady.store(false, std::memory_order_release);

    printf("[*] Stopping GPU processing thread...\n");
    gpuProcessor.stop();

    if (deviceManager)
    {
        deviceManager->stopDevices();
        deviceManager.reset();
    }

    if (upsampler)
    {
        upsampler->shutdown();
        upsampler.reset();
    }

    initialized = false;
    printf("[+] Graceful shutdown complete.\n");
}
