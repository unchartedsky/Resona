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

#include "GpuUpsampler.h"
#include "VulkanUpsampler.h"

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
std::atomic<bool> g_cpuReady{false};
std::atomic<bool> g_gpuReady{false};
std::atomic<uint64_t> g_capturedInputFrames{0};   // Track total captured input frames
std::atomic<uint64_t> g_processedInputFrames{0};  // Track total processed input frames
std::atomic<uint64_t> g_processedOutputFrames{0}; // Track total output frames added to output buffer
std::atomic<uint64_t> g_playedOutputFrames{0};    // Track total output frames played

static std::unique_ptr<GpuUpsampler> g_upsampler;

// === Improved Ring Buffer ===
class FloatRingBuffer
{
  public:
    void init(uint32_t totalSamples)
    {
        buffer.resize(totalSamples);
        size = totalSamples;

        // Check if size is power of 2 for optimal performance
        const bool isPowerOf2 = (size & (size - 1)) == 0;
        if (!isPowerOf2)
        {
            printf("[!] Warning: Ring buffer size is not power of 2, performance may be suboptimal\n");
        }
        else
        {
            // For power-of-2 sizes, we can use fast bitwise AND instead of modulo
            sizeMask = size - 1;
            useFastModulo = true;
            printf("[+] Ring buffer using fast power-of-2 modulo optimization\n");
        }

        printf("[+] Ring buffer initialized: %u samples (%.1f KB)\n", size, (size * sizeof(float)) / 1024.0f);
    }

    void push(const float *data, uint32_t count) noexcept
    {
        if (count > size) [[unlikely]]
        {
            printf("[!] Push count exceeds buffer size\n");
            return;
        }

        const uint32_t writeIdx = writePos.load(std::memory_order_relaxed);

        // OPTIMIZED: Use fast bitwise AND for power-of-2 sizes
        const uint32_t writeOffset = useFastModulo ? (writeIdx & sizeMask) : (writeIdx % size);

        // Prefetch next cache line for better performance
        if (count > 64)
        { // Only for larger transfers
            PREFETCH_WRITE(&buffer[writeOffset]);
        }

        // Check if we can copy to contiguous memory region
        const uint32_t endSpace = size - writeOffset;
        if (count <= endSpace)
        {
            // Single contiguous copy
            std::memcpy(&buffer[writeOffset], data, count * sizeof(float));
        }
        else
        {
            // Split copy across buffer boundary
            std::memcpy(&buffer[writeOffset], data, endSpace * sizeof(float));
            std::memcpy(&buffer[0], data + endSpace, (count - endSpace) * sizeof(float));
        }

        writePos.store(writeIdx + count, std::memory_order_release);
    }

    uint32_t pop(float *out, uint32_t count) noexcept
    {
        const uint32_t readIdx = readPos.load(std::memory_order_relaxed);
        const uint32_t writeIdx = writePos.load(std::memory_order_acquire);
        const uint32_t available = writeIdx - readIdx;
        const uint32_t toRead = std::min(count, available);

        if (toRead == 0) [[unlikely]]
        {
            return 0;
        }

        // OPTIMIZED: Use fast bitwise AND for power-of-2 sizes
        const uint32_t readOffset = useFastModulo ? (readIdx & sizeMask) : (readIdx % size);

        // Prefetch next cache line for better performance
        if (toRead > 64)
        { // Only for larger transfers
            PREFETCH_READ(&buffer[readOffset]);
        }

        // Check if we can copy from contiguous memory region
        const uint32_t endSpace = size - readOffset;
        if (toRead <= endSpace)
        {
            // Single contiguous copy
            std::memcpy(out, &buffer[readOffset], toRead * sizeof(float));
        }
        else
        {
            // Split copy across buffer boundary
            std::memcpy(out, &buffer[readOffset], endSpace * sizeof(float));
            std::memcpy(out + endSpace, &buffer[0], (toRead - endSpace) * sizeof(float));
        }

        readPos.store(readIdx + toRead, std::memory_order_release);
        return toRead;
    }

    uint32_t available() const noexcept
    {
        return writePos.load(std::memory_order_acquire) - readPos.load(std::memory_order_relaxed);
    }

  private:
    std::vector<float> buffer;

    // Cache-line aligned atomics to prevent false sharing
    // Warning C4324: structure was padded due to alignment specifier - this is intentional
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
    alignas(64) std::atomic<uint32_t> writePos{0};
    alignas(64) std::atomic<uint32_t> readPos{0};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    uint32_t size{0};
    uint32_t sizeMask{0}; // For fast power-of-2 modulo (size - 1)
    bool useFastModulo{false};
};

// === Dual Ring Buffers ===
static FloatRingBuffer g_inputRing;  // 44.1kHz captured audio
static FloatRingBuffer g_outputRing; // 384kHz upsampled audio

// === GPU Processing Thread ===
class GpuProcessingThread
{
  public:
    void start()
    {
        if (running)
            return;

        running = true;
        thread = std::thread([this]() {
// Set high priority for GPU processing thread
#ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif

            printf("[+] GPU processing thread started (adaptive mode)\n");

            // Reusable buffer (will grow as needed)
            std::vector<float> inputBuffer;

            while (running)
            {
                if (!g_gpuReady.load(std::memory_order_acquire) || !g_upsampler)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                auto *vulkanUpsampler = static_cast<VulkanUpsampler *>(g_upsampler.get());

                // === ADAPTIVE BATCH SIZING ===
                // Update adaptive parameters based on current buffer state
                const uint32_t outputBufferFrames = g_outputRing.available() / AudioConfig::CHANNELS;

                // REDUCED TARGET: 30% instead of 50% to prevent overgeneration
                // Lower target = less aggressive frame generation = less audio distortion
                const uint32_t targetBufferFrames =
                    AudioConfig::OUTPUT_RING_BUFFER_FRAMES * 30 / 100; // Target 30% full

                vulkanUpsampler->updateAdaptiveParams(outputBufferFrames, targetBufferFrames);

                // Get recommended batch size based on buffer pressure
                const uint32_t batchFrames = vulkanUpsampler->getRecommendedBatchSize();
                const uint32_t batchSamples = batchFrames * AudioConfig::CHANNELS;

                // OPTIMIZATION: Submit multiple batches when buffer is low and slots are available
                const size_t availableSlots = vulkanUpsampler->getAvailableSlots();
                const float bufferRatio = static_cast<float>(outputBufferFrames) / targetBufferFrames;

                // Determine how many batches to submit based on buffer state and available slots
                uint32_t batchesToSubmit = 1;
                if (bufferRatio < 0.3f && availableSlots > 3)
                {
                    // Critical: Submit as many batches as we can (up to 4)
                    batchesToSubmit = std::min(static_cast<uint32_t>(availableSlots), 4u);
                }
                else if (bufferRatio < 0.5f && availableSlots > 2)
                {
                    // Low: Submit 2-3 batches
                    batchesToSubmit = std::min(static_cast<uint32_t>(availableSlots), 3u);
                }
                else if (availableSlots > 1)
                {
                    // Normal: Submit 1-2 batches
                    batchesToSubmit = std::min(static_cast<uint32_t>(availableSlots), 2u);
                }

                // Check if we have enough input data to process all batches
                const uint32_t availableInputFrames = g_inputRing.available() / AudioConfig::CHANNELS;
                const uint32_t totalRequiredFrames = batchFrames * batchesToSubmit;

                if (availableInputFrames < totalRequiredFrames)
                {
                    // Adjust batches to available input
                    batchesToSubmit = std::max(1u, availableInputFrames / batchFrames);

                    if (batchesToSubmit == 0)
                    {
                        // Not enough input data - wait a bit
                        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::GPU_THREAD_SLEEP_MS));
                        continue;
                    }
                }

                // Check if output buffer has space (prevent overflow)
                const uint32_t outputFreeFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES - outputBufferFrames;
                const float ratio =
                    static_cast<float>(AudioConfig::OUTPUT_SAMPLE_RATE) / AudioConfig::INPUT_SAMPLE_RATE;
                const uint32_t expectedOutputFrames = static_cast<uint32_t>(batchFrames * ratio);
                const uint32_t totalExpectedOutput = expectedOutputFrames * batchesToSubmit;

                if (outputFreeFrames < totalExpectedOutput)
                {
                    // Output buffer nearly full - wait for playback to drain it
                    std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::GPU_THREAD_SLEEP_MS));
                    continue;
                }

                // Resize input buffer if needed
                const uint32_t maxBatchSamples = batchSamples * batchesToSubmit;
                if (inputBuffer.size() < maxBatchSamples)
                {
                    inputBuffer.resize(maxBatchSamples);
                }

                // Submit multiple batches
                uint32_t successfulSubmissions = 0;
                for (uint32_t i = 0; i < batchesToSubmit; ++i)
                {
                    // Read input data from input ring buffer
                    const uint32_t readSamples = g_inputRing.pop(inputBuffer.data() + (i * batchSamples), batchSamples);
                    const uint32_t readFrames = readSamples / AudioConfig::CHANNELS;

                    if (readFrames == 0)
                    {
                        break;
                    }

                    // Submit to GPU for processing (async)
                    uint64_t sequenceId = vulkanUpsampler->processAsync(
                        inputBuffer.data() + (i * batchSamples), readFrames,
                        [readFrames](const float *output, uint32_t outputFrames) {
                            // Callback executes when GPU work completes
                            const uint32_t outputSamples = outputFrames * AudioConfig::CHANNELS;
                            g_outputRing.push(output, outputSamples);

                            // Track processing statistics
                            g_processedInputFrames.fetch_add(readFrames, std::memory_order_relaxed);
                            g_processedOutputFrames.fetch_add(outputFrames, std::memory_order_relaxed);
                        });

                    if (sequenceId != 0)
                    {
                        successfulSubmissions++;
                    }
                    else
                    {
                        // Failed to submit - GPU slots busy, stop trying
                        break;
                    }
                }

                // Poll for completed work (non-blocking)
                vulkanUpsampler->tryPollAll();

                // Adaptive sleep: sleep less when buffer is low, more when buffer is high
                uint32_t sleepMs = 0;

                if (bufferRatio < 0.3f)
                {
                    // Critical: No sleep - process as fast as possible
                    sleepMs = 0;
                }
                else if (bufferRatio < 0.6f)
                {
                    // Medium: Short sleep
                    sleepMs = 1;
                }
                else
                {
                    // Healthy: Normal sleep
                    sleepMs = AudioConfig::GPU_THREAD_SLEEP_MS;
                }

                if (sleepMs > 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
                }
            }

            printf("[+] GPU processing thread stopped\n");
        });
    }

    void stop()
    {
        if (!running)
            return;

        running = false;
        if (thread.joinable())
        {
            thread.join();
        }
    }

    ~GpuProcessingThread()
    {
        stop();
    }

  private:
    std::atomic<bool> running{false};
    std::thread thread;
};

static GpuProcessingThread g_gpuProcessor;

// === Signal Handler ===
void signal_handler(int signal)
{
    if (signal == SIGINT)
    {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_running = false;
    }
}

// === RAII Audio Device Manager ===
class AudioDeviceManager
{
  public:
    ~AudioDeviceManager()
    {
        cleanup();
    }

    bool initializeCapture()
    {
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.sampleRate = AudioConfig::INPUT_SAMPLE_RATE;
        config.capture.format = ma_format_f32;
        config.capture.channels = AudioConfig::CHANNELS;
        config.dataCallback = capture_callback;

        // === Performance Optimizations ===

        // Disable miniaudio resampling and conversion (already done, keep these)
        config.noPreSilencedOutputBuffer = MA_TRUE;
        config.noClip = MA_TRUE;
        config.noFixedSizedCallback = MA_TRUE;

        // NEW: Set performance profile for low latency
        config.performanceProfile = ma_performance_profile_low_latency;

// NEW: WASAPI-specific optimizations (Windows only)
#ifdef MA_WIN32
        config.wasapi.noAutoConvertSRC = MA_TRUE;      // Disable WASAPI's sample rate converter
        config.wasapi.noDefaultQualitySRC = MA_TRUE;   // Disable quality SRC for speed
        config.wasapi.noHardwareOffloading = MA_FALSE; // Allow hardware offloading
#endif

        if (ma_device_init(nullptr, &config, &captureDevice) != MA_SUCCESS)
        {
            printf("[!] Failed to initialize capture device\n");
            return false;
        }

// Set high priority for capture thread (Windows)
#ifdef _WIN32
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        HANDLE threadHandle = (HANDLE)captureDevice.thread;
        if (threadHandle && threadHandle != INVALID_HANDLE_VALUE)
        {
            if (!SetThreadPriority(threadHandle, THREAD_PRIORITY_TIME_CRITICAL))
            {
                printf("[*] Warning: Could not set capture thread priority (error: %lu)\n", GetLastError());
            }
            else
            {
                printf("[+] Capture thread priority set to TIME_CRITICAL\n");
            }
        }
#endif

        // Report actual device configuration
        printf("[+] Capture device initialized: %u Hz, %u frames/period, %u periods\n", captureDevice.sampleRate,
               captureDevice.capture.internalPeriodSizeInFrames, captureDevice.capture.internalPeriods);

        // Calculate and report callback frequency
        const float callbackFrequency =
            static_cast<float>(captureDevice.sampleRate) / captureDevice.capture.internalPeriodSizeInFrames;
        const float callbackIntervalMs = 1000.0f / callbackFrequency;

        printf("[+] Capture callback: %.1f Hz (every %.2f ms)\n", callbackFrequency, callbackIntervalMs);

        captureInitialized = true;
        return true;
    }

    bool initializePlayback()
    {
        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.sampleRate = AudioConfig::OUTPUT_SAMPLE_RATE;
        config.playback.format = ma_format_f32;
        config.playback.channels = AudioConfig::CHANNELS;
        config.dataCallback = playback_callback;

        // === Performance Optimizations ===

        // Disable ALL miniaudio resampling and conversion
        config.noPreSilencedOutputBuffer = MA_TRUE;
        config.noClip = MA_TRUE;
        config.noFixedSizedCallback = MA_TRUE; // Allow variable-size callbacks

        // NEW: Set performance profile for low latency
        config.performanceProfile = ma_performance_profile_low_latency;

// NEW: WASAPI-specific optimizations (Windows only)
#ifdef MA_WIN32
        config.wasapi.noAutoConvertSRC = MA_TRUE;      // Disable WASAPI's sample rate converter
        config.wasapi.noDefaultQualitySRC = MA_TRUE;   // Disable quality SRC for speed
        config.wasapi.noHardwareOffloading = MA_FALSE; // Allow hardware offloading

        // NEW: Try to use exclusive mode for lowest latency (may fail, that's OK)
        config.wasapi.usage = ma_wasapi_usage_pro_audio; // Pro audio usage hint
#endif

        if (ma_device_init(nullptr, &config, &playbackDevice) != MA_SUCCESS)
        {
            printf("[!] Failed to initialize playback device\n");
            return false;
        }

        // Report actual device sample rate
        printf("[+] Playback device initialized: %u Hz (requested: %u Hz), %u frames/period, %u periods\n",
               playbackDevice.sampleRate, AudioConfig::OUTPUT_SAMPLE_RATE,
               playbackDevice.playback.internalPeriodSizeInFrames, playbackDevice.playback.internalPeriods);

        playbackInitialized = true;
        return true;
    }

    bool startDevices()
    {
        if (!captureInitialized || !playbackInitialized)
        {
            return false;
        }

        // Start capture device first
        if (ma_device_start(&captureDevice) != MA_SUCCESS)
        {
            printf("[!] Failed to start capture device\n");
            return false;
        }

        // Wait for initial buffer fill
        waitForPrebuffer();

        // Start playback device after prebuffering
        if (ma_device_start(&playbackDevice) != MA_SUCCESS)
        {
            printf("[!] Failed to start playback device\n");
            ma_device_stop(&captureDevice);
            return false;
        }

        return true;
    }

    bool restartPlayback()
    {
        if (!playbackInitialized)
        {
            return false;
        }

        ma_device_stop(&playbackDevice);
        waitForPrebuffer();

        return ma_device_start(&playbackDevice) == MA_SUCCESS;
    }

    void stopDevices()
    {
        printf("[*] Stopping audio devices...\n");

        if (captureInitialized)
        {
            ma_device_stop(&captureDevice);
        }

        if (playbackInitialized)
        {
            ma_device_stop(&playbackDevice);
        }

        // Give callbacks time to exit
        ma_sleep(50);
        printf("[+] Audio devices stopped\n");
    }

    // Forward declarations for callback functions
    static void capture_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount);
    static void playback_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount);

  private:
    ma_device captureDevice{};
    ma_device playbackDevice{};
    bool captureInitialized = false;
    bool playbackInitialized = false;

    void cleanup()
    {
        if (captureInitialized)
        {
            ma_device_uninit(&captureDevice);
        }
        if (playbackInitialized)
        {
            ma_device_uninit(&playbackDevice);
        }
    }

    void waitForPrebuffer()
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
};

// === Improved Audio Callbacks ===
void AudioDeviceManager::capture_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount)
{
    (void)output;
    (void)device;

    // Early exit checks
    if (!g_running.load(std::memory_order_acquire))
    {
        return;
    }

    if (!input) [[unlikely]]
    {
        return;
    }

    const auto *inputSamples = static_cast<const float *>(input);
    const uint32_t samples = frameCount * AudioConfig::CHANNELS;

    // FAST PATH: Just push to input ring buffer (no GPU processing here)
    g_inputRing.push(inputSamples, samples);

    // Track captured frames
    g_capturedInputFrames.fetch_add(frameCount, std::memory_order_relaxed);
}

void AudioDeviceManager::playback_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount)
{
    (void)input;
    (void)device;

    // Check if we're shutting down
    if (!g_running.load(std::memory_order_acquire))
    {
        auto *outputSamples = static_cast<float *>(output);
        const uint32_t requiredSamples = frameCount * AudioConfig::CHANNELS;
        std::memset(outputSamples, 0, requiredSamples * sizeof(float));
        return;
    }

    auto *outputSamples = static_cast<float *>(output);
    const uint32_t requiredSamples = frameCount * AudioConfig::CHANNELS;

    const uint32_t actualSamples = g_outputRing.pop(outputSamples, requiredSamples);

    if (actualSamples < requiredSamples) [[unlikely]]
    {
        // Zero-fill remaining samples
        const size_t remainingBytes = (requiredSamples - actualSamples) * sizeof(float);
        std::memset(outputSamples + actualSamples, 0, remainingBytes);

        // Only trigger underrun if buffer is critically low
        const uint32_t remainingFrames = g_outputRing.available() / AudioConfig::CHANNELS;
        if (remainingFrames < AudioConfig::MIN_BUFFER_FRAMES)
        {
            g_underrun.store(true, std::memory_order_relaxed);
        }
    }

    // Track played frames
    g_playedOutputFrames.fetch_add(frameCount, std::memory_order_relaxed);
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
            const uint32_t batchSize = vulkanUpsampler->getRecommendedBatchSize();
            const float currentRatio = vulkanUpsampler->getCurrentRatio();

            // Calculate buffer fill rates (frames per second)
            const int32_t outputBufferDelta =
                static_cast<int32_t>(outputBufferFrames) - static_cast<int32_t>(lastOutputBufferLevel);
            const double outputFillRate = (outputBufferDelta * 1000.0) / elapsedMs;

            // Calculate buffer pressure (based on dynamic target level)
            const uint32_t targetBufferFrames = vulkanUpsampler->getTargetBufferLevel();
            float bufferPressure = 1.0f;

            if (targetBufferFrames > 0)
            {
                bufferPressure = std::min(1.0f, static_cast<float>(outputBufferFrames) / targetBufferFrames);
            }

            // Calculate ratio deviation from base
            const float baseRatio = static_cast<float>(AudioConfig::OUTPUT_SAMPLE_RATE) /
                                    static_cast<float>(AudioConfig::INPUT_SAMPLE_RATE);
            const float ratioDeviation = ((currentRatio / baseRatio) - 1.0f) * 100.0f; // Percentage

            // Calculate total runtime
            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

            // SIMPLIFIED: Single-line compact status
            printf("\r\033[K"); // Clear line
            printf("Out:%u(%.0f%%%+.0f) GPU:%u/%u Batch:%u Ratio:%.6f(%+.3f%%) Press:%.0f%% %llus", outputBufferFrames,
                   (outputBufferFrames * 100.0f) / AudioConfig::OUTPUT_RING_BUFFER_FRAMES, outputFillRate,
                   static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS - availableSlots),
                   static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS), batchSize, currentRatio, ratioDeviation,
                   bufferPressure * 100.0f, static_cast<unsigned long long>(totalElapsed));
            fflush(stdout);

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

    if (totalElapsed > 0)
    {
        printf("[+] Session statistics:\n");
        printf("    Runtime: %lld seconds\n", totalElapsed);
        printf("    Captured: %llu frames (%.1f fps avg)\n", totalCaptured,
               totalCaptured / static_cast<double>(totalElapsed));
        printf("    Processed: %llu frames (%.1f fps avg)\n", totalProcessed,
               totalProcessed / static_cast<double>(totalElapsed));
        printf("    Played: %llu frames (%.1f fps avg)\n", totalPlayed,
               totalPlayed / static_cast<double>(totalElapsed));
        printf("    Loss rate: %.3f%%\n",
               totalCaptured > 0 ? ((totalCaptured - totalProcessed) * 100.0 / totalCaptured) : 0.0);
    }
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
    g_gpuReady.store(true, std::memory_order_release);

    // Step 5: Start GPU processing thread
    printf("[*] Starting GPU processing thread...\n");
    g_gpuProcessor.start();

    printf("[*] Initializing audio devices...\n");

    // Step 6: Initialize audio devices
    AudioDeviceManager deviceManager;
    if (!deviceManager.initializeCapture() || !deviceManager.initializePlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        return EXIT_FAILURE;
    }

    // Step 7: Start devices
    if (!deviceManager.startDevices())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        return EXIT_FAILURE;
    }

    // Step 8: Reset adaptive target now that prebuffer is complete and stable
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
