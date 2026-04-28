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

#include "Audio/AudioDeviceManager.h"
#include "Audio/AudioCallbackContext.h"
#include "Audio/FloatRingBuffer.h"
#include "GpuUpsampler.h"
#include "RenderPipeline/GpuProcessingThread.h"
#include "VulkanUpsampler.h"
#include "Runtime/StatusReporter.h"

// === Configuration Constants ===
namespace AudioConfig
{
static constexpr uint32_t INPUT_SAMPLE_RATE = 44100;
static constexpr uint32_t OUTPUT_SAMPLE_RATE = 384000;
static constexpr uint32_t CHANNELS = 2;

// DUAL BUFFER ARCHITECTURE:
// Input buffer: Stores captured 44.1kHz audio (fast writes from capture callback)
// Output buffer: Stores upsampled 384kHz audio (reads from playback callback)

// OPTIMIZED: Power-of-2 buffer sizes for optimal modulo performance
// Input buffer: 2 seconds @ 44.1kHz = ~88200 frames -> round up to 131072 (2^17)
static constexpr uint32_t INPUT_RING_BUFFER_FRAMES = 131072; // Power of 2 (2^17 = 131072 frames)

// Output buffer: 2 seconds @ 384kHz = ~768000 frames -> round up to 1048576 (2^20)
static constexpr uint32_t OUTPUT_RING_BUFFER_FRAMES = 1048576; // Power of 2 (2^20 = 1048576 frames)

// OPTIMIZED: Increased prebuffer to 400ms for stable startup (prevent initial underrun)
static constexpr uint32_t PREBUFFER_FRAMES = OUTPUT_SAMPLE_RATE * 40 / 100; // 400ms @ 384kHz

// CRITICAL: Safety threshold to prevent underruns
static constexpr uint32_t MIN_BUFFER_FRAMES = OUTPUT_SAMPLE_RATE * 25 / 100; // 250ms @ 384kHz

static constexpr uint32_t SLEEP_INTERVAL_MS = 5;
static constexpr uint32_t MAIN_LOOP_SLEEP_MS = 10;

// GPU processing thread sleep interval when no work available
static constexpr uint32_t GPU_THREAD_SLEEP_MS = 1;

} // namespace AudioConfig

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

static std::unique_ptr<GpuUpsampler> g_upsampler;

// === Dual Ring Buffers ===
static FloatRingBuffer g_inputRing;  // 44.1kHz captured audio
static FloatRingBuffer g_outputRing; // 384kHz upsampled audio

static GpuProcessingContext g_gpuProcessingContext;
static GpuProcessingThread g_gpuProcessor(&g_gpuProcessingContext);

static void waitForOutputPrebuffer()
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

    SessionStatistics sessionStatistics{};
    sessionStatistics.totalElapsedSeconds = totalElapsed;
    sessionStatistics.totalCapturedFrames = totalCaptured;
    sessionStatistics.totalProcessedFrames = totalProcessed;
    sessionStatistics.totalPlayedFrames = totalPlayed;
    StatusReporter::PrintSessionStatistics(sessionStatistics);
}

// === Main Application ===
int main()
{
    std::signal(SIGINT, signal_handler);

    printf("[*] Initializing GPU upsampler...\n");

    // Step 1: Initialize GPU upsampler
    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE,
                                 AudioConfig::CHANNELS))
    {
        printf("[!] Failed to initialize upsampler\n");
        return EXIT_FAILURE;
    }

    // Step 2: Load shader - use Nearest for noise-free resampling
    g_upsampler->setKernel(ResampleKernel::Nearest);

    // Step 3: Initialize dual ring buffers
    printf("[*] Initializing dual ring buffers...\n");
    g_inputRing.init(AudioConfig::INPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    g_outputRing.init(AudioConfig::OUTPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);

    // Step 4: Mark GPU as ready
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

    // Step 5: Start GPU processing thread
    printf("[*] Starting GPU processing thread...\n");
    g_gpuProcessor.start();

    printf("[*] Initializing audio devices...\n");

    // Step 6: Initialize audio devices
    AudioCallbackContext audioCallbackContext{};
    audioCallbackContext.running = &g_running;
    audioCallbackContext.underrun = &g_underrun;
    audioCallbackContext.inputRing = &g_inputRing;
    audioCallbackContext.outputRing = &g_outputRing;
    audioCallbackContext.capturedInputFrames = &g_capturedInputFrames;
    audioCallbackContext.playedOutputFrames = &g_playedOutputFrames;
    audioCallbackContext.channels = AudioConfig::CHANNELS;
    audioCallbackContext.minBufferFrames = AudioConfig::MIN_BUFFER_FRAMES;

    AudioDeviceManager deviceManager(&audioCallbackContext);
    if (!deviceManager.initializeCapture() || !deviceManager.initializePlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        return EXIT_FAILURE;
    }

    // Step 7: Start capture first so input/output generation can begin
    if (!deviceManager.startCapture())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        return EXIT_FAILURE;
    }

    // Step 8: Wait for output prebuffer before enabling playback
    waitForOutputPrebuffer();

    if (!deviceManager.startPlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager.stopDevices();
        return EXIT_FAILURE;
    }

    // Step 9: Reset adaptive target now that prebuffer is complete and stable
    // This ensures we capture the full buffer level (e.g. 15%) as the target, not the initial empty state
    if (g_upsampler)
    {
        static_cast<VulkanUpsampler *>(g_upsampler.get())->resetAdaptiveTarget();
    }

    printf("[+] Real-time GPU upsampler started (%u -> %uHz)\n", AudioConfig::INPUT_SAMPLE_RATE,
           AudioConfig::OUTPUT_SAMPLE_RATE);
    printf("[*] Architecture: Capture -> Input Buffer (44.1kHz) -> GPU -> Output Buffer (384kHz) -> Playback\n");
    printf("[*] Press Ctrl+C to stop...\n");

    // Run main processing loop
    runMainLoop(deviceManager);

    // Graceful shutdown sequence
    printf("[*] Initiating shutdown sequence...\n");

    // Mark runtime as stopping so callbacks and other runtime paths observe shutdown consistently.
    g_running.store(false, std::memory_order_release);

    // Mark GPU as not ready
    g_gpuReady.store(false, std::memory_order_release);

    // Step 1: Stop GPU processing thread
    printf("[*] Stopping GPU processing thread...\n");
    g_gpuProcessor.stop();

    // Step 2: Stop audio devices
    deviceManager.stopDevices();

    // Step 3: Cleanup GPU resources
    if (g_upsampler)
    {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    printf("[+] Graceful shutdown complete.\n");
    return EXIT_SUCCESS;
}
