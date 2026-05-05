#include "AppRuntime.h"

#include "../Audio/AudioConfig.h"
#include "StatusReporter.h"
#include "RuntimeHooks.h"
#include "../VulkanUpsampler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

void AppRuntime::waitForOutputPrebuffer() const
{
    const uint32_t targetFrames = AudioConfig::PREBUFFER_FRAMES;
    const uint32_t safetyMargin = targetFrames / 20;
    const uint32_t safeTargetFrames = targetFrames + safetyMargin;

    printf("[*] Waiting for prebuffer (target: %u frames + %u margin)...\n", targetFrames, safetyMargin);

    while (g_outputRing.available() / AudioConfig::CHANNELS < safeTargetFrames)
    {
        const uint32_t currentFrames = g_outputRing.available() / AudioConfig::CHANNELS;
        printf("    Progress: %u/%u frames (%.1f%%)     \r", currentFrames, safeTargetFrames,
               (currentFrames * 100.0f) / safeTargetFrames);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::SLEEP_INTERVAL_MS));
    }

    printf("\n[+] Prebuffer complete (filled: %u frames)                    \n",
           g_outputRing.available() / AudioConfig::CHANNELS);
}

void AppRuntime::runMainLoop()
{
    auto lastStatusTime = std::chrono::steady_clock::now();
    auto startTime = std::chrono::steady_clock::now();
    uint32_t lastOutputBufferLevel = 0;

    while (g_running)
    {
        if (g_underrun.exchange(false))
        {
            printf("\r%*s\r", 120, "");
            printf("[!] Underrun detected. Re-buffering...\n");

            g_restartRequested.store(true, std::memory_order_release);
        }

        if (g_restartRequested.load(std::memory_order_acquire) &&
            !g_restartInProgress.exchange(true, std::memory_order_acq_rel))
        {
            g_restartRequested.store(false, std::memory_order_release);

            if (deviceManager && deviceManager->restartPlayback())
            {
                printf("[+] Playback resumed after underrun\n");
                g_restartInProgress.store(false, std::memory_order_release);
            }
            else
            {
                printf("[!] Failed to restart playback after underrun\n");
                g_running.store(false, std::memory_order_release);
                g_restartInProgress.store(false, std::memory_order_release);
                break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatusTime).count();

        if (elapsedMs >= 500)
        {
            const uint32_t outputBufferFrames = g_outputRing.available() / AudioConfig::CHANNELS;

            const auto *vulkanUpsampler = static_cast<VulkanUpsampler *>(g_upsampler.get());
            const size_t availableSlots = vulkanUpsampler->getAvailableSlots();
            const float currentRatio = vulkanUpsampler->getCurrentRatio();

            const int32_t outputBufferDelta =
                static_cast<int32_t>(outputBufferFrames) - static_cast<int32_t>(lastOutputBufferLevel);
            const double outputFillRate = (outputBufferDelta * 1000.0) / elapsedMs;

            const uint32_t targetBufferFrames = vulkanUpsampler->getTargetBufferLevel();
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
            statusSnapshot.busySlots = static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS - availableSlots);
            statusSnapshot.totalSlots = static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS);
            statusSnapshot.currentRatio = currentRatio;
            statusSnapshot.baseRatio = baseRatio;
            statusSnapshot.targetFillRatio = bufferPressure;
            statusSnapshot.outputRingCapacityFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES;
            const uint32_t minObservedOutputFrames = g_minObservedOutputFrames.load(std::memory_order_relaxed);
            statusSnapshot.minObservedOutputFrames =
                minObservedOutputFrames == UINT32_MAX ? outputBufferFrames : minObservedOutputFrames;
            statusSnapshot.zeroFillEvents = g_zeroFillEvents.load(std::memory_order_relaxed);
            statusSnapshot.zeroFillSamples = g_zeroFillSamples.load(std::memory_order_relaxed);
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
    const uint64_t totalCaptured = g_capturedInputFrames.load(std::memory_order_relaxed);
    const uint64_t totalProcessed = g_processedInputFrames.load(std::memory_order_relaxed);
    const uint64_t totalPlayed = g_playedOutputFrames.load(std::memory_order_relaxed);
    const uint32_t minObservedOutputFrames = g_minObservedOutputFrames.load(std::memory_order_relaxed);

    SessionStatistics sessionStatistics{};
    sessionStatistics.totalElapsedSeconds = totalElapsed;
    sessionStatistics.totalCapturedFrames = totalCaptured;
    sessionStatistics.totalProcessedFrames = totalProcessed;
    sessionStatistics.totalPlayedFrames = totalPlayed;
    sessionStatistics.minObservedOutputFrames =
        minObservedOutputFrames == UINT32_MAX ? 0 : minObservedOutputFrames;
    sessionStatistics.zeroFillEvents = g_zeroFillEvents.load(std::memory_order_relaxed);
    sessionStatistics.zeroFillSamples = g_zeroFillSamples.load(std::memory_order_relaxed);
    StatusReporter::PrintSessionStatistics(sessionStatistics);
}

bool AppRuntime::Initialize()
{
    printf("[*] Initializing GPU upsampler...\n");

    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE, AudioConfig::CHANNELS))
    {
        printf("[!] Failed to initialize upsampler\n");
        return false;
    }

    g_upsampler->setKernel(ResampleKernel::Nearest);

    printf("[*] Initializing dual ring buffers...\n");
    g_inputRing.init(AudioConfig::INPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    g_outputRing.init(AudioConfig::OUTPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    g_zeroFillEvents.store(0, std::memory_order_relaxed);
    g_zeroFillSamples.store(0, std::memory_order_relaxed);
    g_minObservedOutputFrames.store(AudioConfig::OUTPUT_RING_BUFFER_FRAMES, std::memory_order_relaxed);

    g_gpuProcessingContext.gpuReady = &g_gpuReady;
    g_gpuProcessingContext.upsampler = static_cast<VulkanUpsampler *>(g_upsampler.get());
    g_gpuProcessingContext.inputRing = &g_inputRing;
    g_gpuProcessingContext.outputRing = &g_outputRing;
    g_gpuProcessingContext.processedInputFrames = &g_processedInputFrames;
    g_gpuProcessingContext.processedOutputFrames = &g_processedOutputFrames;
    g_gpuProcessingContext.inputSampleRate = AudioConfig::INPUT_SAMPLE_RATE;
    g_gpuProcessingContext.outputSampleRate = AudioConfig::OUTPUT_SAMPLE_RATE;
    g_gpuProcessingContext.channels = AudioConfig::CHANNELS;
    g_gpuProcessingContext.outputRingCapacityFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES;
    g_gpuProcessingContext.idleSleepMs = AudioConfig::GPU_THREAD_SLEEP_MS;

    g_gpuReady.store(true, std::memory_order_release);

    printf("[*] Starting GPU processing thread...\n");
    g_gpuProcessor.start();

    printf("[*] Initializing audio devices...\n");
    audioCallbackContext.running = &g_running;
    audioCallbackContext.underrun = &g_underrun;
    audioCallbackContext.inputRing = &g_inputRing;
    audioCallbackContext.outputRing = &g_outputRing;
    audioCallbackContext.capturedInputFrames = &g_capturedInputFrames;
    audioCallbackContext.playedOutputFrames = &g_playedOutputFrames;
    audioCallbackContext.zeroFillEvents = &g_zeroFillEvents;
    audioCallbackContext.zeroFillSamples = &g_zeroFillSamples;
    audioCallbackContext.minObservedOutputFrames = &g_minObservedOutputFrames;
    audioCallbackContext.channels = AudioConfig::CHANNELS;
    audioCallbackContext.minBufferFrames = AudioConfig::MIN_BUFFER_FRAMES;

    deviceManager = std::make_unique<AudioDeviceManager>(&audioCallbackContext);
    if (!deviceManager->initializeCapture() || !deviceManager->initializePlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager.reset();
        return false;
    }

    if (!deviceManager->startCapture())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager.reset();
        return false;
    }

    waitForOutputPrebuffer();
    g_minObservedOutputFrames.store(g_outputRing.available() / AudioConfig::CHANNELS, std::memory_order_relaxed);

    if (!deviceManager->startPlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager->stopDevices();
        deviceManager.reset();
        return false;
    }

    if (g_upsampler)
    {
        static_cast<VulkanUpsampler *>(g_upsampler.get())->resetAdaptiveTarget();
    }

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

void AppRuntime::Shutdown()
{
    if (!initialized)
    {
        return;
    }

    printf("[*] Initiating shutdown sequence...\n");

    g_running.store(false, std::memory_order_release);
    g_gpuReady.store(false, std::memory_order_release);

    printf("[*] Stopping GPU processing thread...\n");
    g_gpuProcessor.stop();

    if (deviceManager)
    {
        deviceManager->stopDevices();
        deviceManager.reset();
    }

    if (g_upsampler)
    {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    initialized = false;
    printf("[+] Graceful shutdown complete.\n");
}
