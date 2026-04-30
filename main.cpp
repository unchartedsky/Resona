#define NOMINMAX

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

// Compiler-specific prefetch intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#define PREFETCH_READ(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#define PREFETCH_WRITE(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
#define PREFETCH_READ(addr) ((void)0)
#define PREFETCH_WRITE(addr) ((void)0)
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "Audio/AudioConfig.h"
#include "Audio/AudioDeviceManager.h"
#include "Audio/AudioCallbackContext.h"
#include "Audio/FloatRingBuffer.h"
#include "GpuUpsampler.h"
#include "RenderPipeline/GpuProcessingThread.h"
#include "Runtime/AppRuntime.h"
#include "Runtime/RuntimeHooks.h"
#include "VulkanUpsampler.h"
#include "Runtime/StatusReporter.h"

// === Global State ===
std::atomic<bool> g_running{true};
std::atomic<bool> g_underrun{false};
std::atomic<bool> g_restartRequested{false};
std::atomic<bool> g_restartInProgress{false};
std::atomic<bool> g_gpuReady{false};
std::atomic<uint64_t> g_capturedInputFrames{0};   // Track total captured input frames
std::atomic<uint64_t> g_processedInputFrames{0};  // Track total processed input frames
std::atomic<uint64_t> g_processedOutputFrames{0}; // Track total output frames added to output buffer
std::atomic<uint64_t> g_playedOutputFrames{0};    // Track total output frames played
std::atomic<uint64_t> g_zeroFillEvents{0};        // Track playback callbacks that had to zero-fill
std::atomic<uint64_t> g_zeroFillSamples{0};       // Track total zero-filled samples
std::atomic<uint32_t> g_minObservedOutputFrames{UINT32_MAX}; // Track worst-case output buffer depth

std::unique_ptr<GpuUpsampler> g_upsampler;

// === Dual Ring Buffers ===
FloatRingBuffer g_inputRing;  // 44.1kHz captured audio
FloatRingBuffer g_outputRing; // 384kHz upsampled audio

GpuProcessingContext g_gpuProcessingContext;
GpuProcessingThread g_gpuProcessor(&g_gpuProcessingContext);

void waitForOutputPrebuffer()
{
    // Add 5% safety margin to prevent initial underrun
    const uint32_t targetFrames = AudioConfig::PREBUFFER_FRAMES;
    const uint32_t safetyMargin = targetFrames / 20; // 5% margin
    const uint32_t safeTargetFrames = targetFrames + safetyMargin;

    printf("[*] Waiting for prebuffer (target: %u frames + %u margin)...\n", targetFrames, safetyMargin);

    while (g_outputRing.available() / AudioConfig::CHANNELS < safeTargetFrames)
    {
        const uint32_t currentFrames = g_outputRing.available() / AudioConfig::CHANNELS;
        printf("    Progress: %u/%u frames (%.1f%%)     \r", currentFrames, safeTargetFrames,
               (currentFrames * 100.0f) / safeTargetFrames);
        fflush(stdout);
        ma_sleep(AudioConfig::SLEEP_INTERVAL_MS);
    }

    printf("\n[+] Prebuffer complete (filled: %u frames)                    \n",
           g_outputRing.available() / AudioConfig::CHANNELS);
}

// === Signal Handler ===
void signal_handler(int signal)
{
    if (signal == SIGINT)
    {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_running = false;
    }
}

// === Main Processing Loop ===
void runMainLoop(AudioDeviceManager &deviceManager)
{
    auto lastStatusTime = std::chrono::steady_clock::now();
    auto startTime = std::chrono::steady_clock::now();
    uint32_t lastInputBufferLevel = 0;
    uint32_t lastOutputBufferLevel = 0;

    while (g_running)
    {
        // Non-blocking underrun check
        if (g_underrun.exchange(false))
        {
            // Clear current line first, then print underrun message
            printf("\r%*s\r", 120, ""); // Clear line with spaces
            printf("[!] Underrun detected. Re-buffering...\n");

            // Request a restart to be handled from the main loop.
            // This avoids detached threads outliving deviceManager during shutdown.
            g_restartRequested.store(true, std::memory_order_release);
        }

        if (g_restartRequested.load(std::memory_order_acquire) &&
            !g_restartInProgress.exchange(true, std::memory_order_acq_rel))
        {
            g_restartRequested.store(false, std::memory_order_release);

            if (deviceManager.restartPlayback())
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

        // Non-blocking status report (every ~500ms)
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatusTime).count();

        if (elapsedMs >= 500)
        {
            const uint32_t inputBufferFrames = g_inputRing.available() / AudioConfig::CHANNELS;
            const uint32_t outputBufferFrames = g_outputRing.available() / AudioConfig::CHANNELS;

            const auto *vulkanUpsampler = static_cast<VulkanUpsampler *>(g_upsampler.get());
            const size_t availableSlots = vulkanUpsampler->getAvailableSlots();
            const float currentRatio = vulkanUpsampler->getCurrentRatio();

            // Calculate buffer fill rates (frames per second)
            const int32_t outputBufferDelta =
                static_cast<int32_t>(outputBufferFrames) - static_cast<int32_t>(lastOutputBufferLevel);
            const double outputFillRate = (outputBufferDelta * 1000.0) / elapsedMs;

            // Calculate buffer pressure against the current adaptive target level.
            const uint32_t targetBufferFrames = vulkanUpsampler->getTargetBufferLevel();
            float bufferPressure = 1.0f;

            if (targetBufferFrames > 0)
            {
                bufferPressure = std::min(1.0f, static_cast<float>(outputBufferFrames) / targetBufferFrames);
            }

            const float baseRatio = static_cast<float>(AudioConfig::OUTPUT_SAMPLE_RATE) /
                                    static_cast<float>(AudioConfig::INPUT_SAMPLE_RATE);

            // Calculate total runtime
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

            lastInputBufferLevel = inputBufferFrames;
            lastOutputBufferLevel = outputBufferFrames;
            lastStatusTime = now;
        }

        // Very short sleep - let other threads run
        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::MAIN_LOOP_SLEEP_MS));
    }
    printf("\n");

    // Print final statistics
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

int main()
{
    std::signal(SIGINT, signal_handler);

    AppRuntime appRuntime;
    if (!appRuntime.Initialize())
    {
        return EXIT_FAILURE;
    }

    const int exitCode = appRuntime.Run();
    appRuntime.Shutdown();
    return exitCode;
}
