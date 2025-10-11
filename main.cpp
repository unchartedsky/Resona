#define NOMINMAX

#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <thread>
#include <chrono>

// Compiler-specific prefetch intrinsics
#if defined(_MSC_VER)
    #include <intrin.h>
    #define PREFETCH_READ(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_READ(addr)  __builtin_prefetch((addr), 0, 3)
    #define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
    #define PREFETCH_READ(addr)  ((void)0)
    #define PREFETCH_WRITE(addr) ((void)0)
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "GpuUpsampler.h"
#include "VulkanUpsampler.h"

// === Configuration Constants ===
namespace AudioConfig {
    static constexpr uint32_t INPUT_SAMPLE_RATE = 44100;
    static constexpr uint32_t OUTPUT_SAMPLE_RATE = 384000;
    static constexpr uint32_t CHANNELS = 2;
    
    // OPTIMIZED: Request small period for lower latency (driver may override)
    // Windows typically enforces 10ms (441 frames @ 44.1kHz) regardless of request
    static constexpr uint32_t CAPTURE_PERIOD_SIZE = 64;   // Requested, not guaranteed
    static constexpr uint32_t CAPTURE_PERIODS = 3;        // Driver default (most common)
    
    // OPTIMIZED: Reduced prebuffer for faster startup
    // 250ms is sufficient for most scenarios, much faster than 500ms
    static constexpr uint32_t PREBUFFER_SAMPLES = OUTPUT_SAMPLE_RATE * 25 / 100; // 250ms

    // CRITICAL: Higher safety threshold to prevent underruns
    static constexpr uint32_t MIN_BUFFER_SAMPLES = OUTPUT_SAMPLE_RATE * 25 / 100; // 250ms (increased from 150ms)

    static constexpr uint32_t SLEEP_INTERVAL_MS = 5;
    
    // Main loop sleep interval - short sleep to remain responsive while yielding CPU
    static constexpr uint32_t MAIN_LOOP_SLEEP_MS = 10;

    // CRITICAL: Ring buffer for stability
    // 2^20 = 1,048,576 samples = ~2.73 seconds @ 384kHz (4MB RAM)
    // This handles ~32 minutes of 1400 samples/sec drift
    static constexpr uint32_t RING_BUFFER_SAMPLES = 1048576;
}

// === Global State ===
std::atomic<bool> g_running{ true };
std::atomic<bool> g_underrun{ false };
std::atomic<bool> g_gpuReady{ false };
std::atomic<uint64_t> g_droppedFrames{ 0 }; // Track frames dropped due to GPU slot exhaustion
std::atomic<uint64_t> g_processedInputFrames{ 0 }; // Track total processed input frames
std::atomic<uint64_t> g_processedOutputSamples{ 0 }; // Track total output samples added to ring buffer

static std::unique_ptr<GpuUpsampler> g_upsampler;

// === GPU Polling Thread ===
class GpuPollerThread {
public:
    void start() {
        if (running) return;
        
        running = true;
        thread = std::thread([this]() {
            // Set high priority for polling thread
            #ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            #endif

            uint32_t spinCount = 0;
            
            while (running) {
                if (!g_gpuReady.load(std::memory_order_acquire) || !g_upsampler) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                
                auto* vulkanUpsampler = static_cast<VulkanUpsampler*>(g_upsampler.get());
                
                // Ultra-aggressive polling with spinlock behavior
                size_t completed = vulkanUpsampler->tryPollAll();

                if (completed > 0) {
                    spinCount = 0;
                    // Continue immediately - don't sleep when there's work
                    continue;
                }
                
                // No work completed - only yield CPU briefly to prevent 100% usage
                ++spinCount;
                
                // ULTRA-OPTIMIZED: Pure spinlock for first 1000 iterations
                // Only yield CPU every 1000 iterations to keep latency minimal
                if (spinCount % 1000 == 0) {
                    std::this_thread::yield();
                    
                    // After many spins, add tiny sleep to prevent CPU saturation
                    if (spinCount > 10000) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
            }
        });
    }
    
    void stop() {
        if (!running) return;
        
        running = false;
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    ~GpuPollerThread() {
        stop();
    }
    
private:
    std::atomic<bool> running{false};
    std::thread thread;
};

static GpuPollerThread g_poller;

// === Signal Handler ===
void signal_handler(int signal) {
    if (signal == SIGINT) {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_running = false;
    }
}

// === Improved Ring Buffer ===
class FloatRingBuffer {
public:
    void init(uint32_t totalSamples) {
        buffer.resize(totalSamples);
        size = totalSamples;
        
        // Check if size is power of 2 for optimal performance
        if ((size & (size - 1)) != 0) {
            printf("[!] Warning: Ring buffer size is not power of 2, performance may be suboptimal\n");
        }
        
        printf("[+] Ring buffer initialized: %u samples (%.1f KB)\n", 
               size, (size * sizeof(float)) / 1024.0f);
    }

    void push(const float* data, uint32_t count) noexcept {
        if (count > size) [[unlikely]] {
            printf("[!] Push count exceeds buffer size\n");
            return;
        }
        
        const uint32_t writeIdx = writePos.load(std::memory_order_relaxed);
        const uint32_t writeOffset = writeIdx % size;
        
        // Prefetch next cache line for better performance
        if (count > 64) { // Only for larger transfers
            PREFETCH_WRITE(&buffer[writeOffset]);
        }
        
        // Check if we can copy to contiguous memory region
        const uint32_t endSpace = size - writeOffset;
        if (count <= endSpace) {
            // Single contiguous copy
            std::memcpy(&buffer[writeOffset], data, count * sizeof(float));
        } else {
            // Split copy across buffer boundary
            std::memcpy(&buffer[writeOffset], data, endSpace * sizeof(float));
            std::memcpy(&buffer[0], data + endSpace, (count - endSpace) * sizeof(float));
        }
        
        writePos.store(writeIdx + count, std::memory_order_release);
    }

    uint32_t pop(float* out, uint32_t count) noexcept {
        const uint32_t readIdx = readPos.load(std::memory_order_relaxed);
        const uint32_t writeIdx = writePos.load(std::memory_order_acquire);
        const uint32_t available = writeIdx - readIdx;
        const uint32_t toRead = std::min(count, available);

        if (toRead == 0) [[unlikely]] {
            return 0;
        }

        const uint32_t readOffset = readIdx % size;
        
        // Prefetch next cache line for better performance
        if (toRead > 64) { // Only for larger transfers
            PREFETCH_READ(&buffer[readOffset]);
        }

        // Check if we can copy from contiguous memory region
        const uint32_t endSpace = size - readOffset;
        if (toRead <= endSpace) {
            // Single contiguous copy
            std::memcpy(out, &buffer[readOffset], toRead * sizeof(float));
        } else {
            // Split copy across buffer boundary
            std::memcpy(out, &buffer[readOffset], endSpace * sizeof(float));
            std::memcpy(out + endSpace, &buffer[0], (toRead - endSpace) * sizeof(float));
        }

        readPos.store(readIdx + toRead, std::memory_order_release);
        return toRead;
    }

    uint32_t available() const noexcept {
        return writePos.load(std::memory_order_acquire) - readPos.load(std::memory_order_relaxed);
    }

private:
    std::vector<float> buffer;
    std::atomic<uint32_t> writePos{0};
    std::atomic<uint32_t> readPos{0};
    uint32_t size{0};
};

static FloatRingBuffer g_ring;

// === RAII Audio Device Manager ===
class AudioDeviceManager {
public:
    ~AudioDeviceManager() {
        cleanup();
    }
    
    bool initializeCapture() {
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.sampleRate = AudioConfig::INPUT_SAMPLE_RATE;
        config.capture.format = ma_format_f32;
        config.capture.channels = AudioConfig::CHANNELS;
        config.dataCallback = capture_callback;

        // CRITICAL: Disable ALL miniaudio resampling and conversion
        config.noPreSilencedOutputBuffer = MA_TRUE;
        config.noClip = MA_TRUE;
        config.noFixedSizedCallback = MA_TRUE;  // Allow variable-size callbacks from hardware

        // Request low-latency period size (driver will use closest supported value)
        config.periodSizeInFrames = AudioConfig::CAPTURE_PERIOD_SIZE;
        config.periods = AudioConfig::CAPTURE_PERIODS;
        
        // Request low-latency/performance mode
        config.performanceProfile = ma_performance_profile_low_latency;

        if (ma_device_init(nullptr, &config, &captureDevice) != MA_SUCCESS) {
            printf("[!] Failed to initialize capture device\n");
            return false;
        }
        
        // Set high priority for capture thread (Windows)
        #ifdef _WIN32
        // Wait a moment for thread to be created
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        HANDLE threadHandle = (HANDLE)captureDevice.thread;
        if (threadHandle && threadHandle != INVALID_HANDLE_VALUE) {
            if (!SetThreadPriority(threadHandle, THREAD_PRIORITY_TIME_CRITICAL)) {
                printf("[*] Warning: Could not set capture thread priority (error: %lu)\n", GetLastError());
            } else {
                printf("[+] Capture thread priority set to TIME_CRITICAL\n");
            }
        }
        #endif
        
        // Report actual device configuration
        printf("[+] Capture device initialized: %u Hz, %u frames/period, %u periods\n",
               captureDevice.sampleRate,
               captureDevice.capture.internalPeriodSizeInFrames,
               captureDevice.capture.internalPeriods);
        
        // Calculate and report callback frequency
        const float callbackFrequency = static_cast<float>(captureDevice.sampleRate) / 
                                       captureDevice.capture.internalPeriodSizeInFrames;
        const float callbackIntervalMs = 1000.0f / callbackFrequency;
        
        printf("[+] Capture callback: %.1f Hz (every %.2f ms)\n", 
               callbackFrequency, callbackIntervalMs);
        
        // Provide helpful information if period size is larger than requested
        if (captureDevice.capture.internalPeriodSizeInFrames > AudioConfig::CAPTURE_PERIOD_SIZE * 2) {
            printf("[*] Note: Driver enforced period size (%u frames) is larger than requested (%u frames)\n",
                   captureDevice.capture.internalPeriodSizeInFrames,
                   AudioConfig::CAPTURE_PERIOD_SIZE);
            printf("[*] This is normal for Windows audio. For lower latency, consider ASIO drivers.\n");
        }
        
        captureInitialized = true;
        return true;
    }
    
    bool initializePlayback() {
        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.sampleRate = AudioConfig::OUTPUT_SAMPLE_RATE;
        config.playback.format = ma_format_f32;
        config.playback.channels = AudioConfig::CHANNELS;
        config.dataCallback = playback_callback;

        // CRITICAL: Disable ALL miniaudio resampling and conversion
        config.noPreSilencedOutputBuffer = MA_TRUE;
        config.noClip = MA_TRUE;
        config.noFixedSizedCallback = MA_TRUE;  // Allow variable-size callbacks
        
        if (ma_device_init(nullptr, &config, &playbackDevice) != MA_SUCCESS) {
            printf("[!] Failed to initialize playback device\n");
            return false;
        }
        
        // Report actual device sample rate
        printf("[+] Playback device initialized: %u Hz (requested: %u Hz), %u frames/period, %u periods\n",
               playbackDevice.sampleRate,
               AudioConfig::OUTPUT_SAMPLE_RATE,
               playbackDevice.playback.internalPeriodSizeInFrames,
               playbackDevice.playback.internalPeriods);
        
        playbackInitialized = true;
        return true;
    }
    
    bool startDevices() {
        if (!captureInitialized || !playbackInitialized) {
            return false;
        }
        
        // Start capture device first
        if (ma_device_start(&captureDevice) != MA_SUCCESS) {
            printf("[!] Failed to start capture device\n");
            return false;
        }
        
        // Wait for initial buffer fill
        waitForPrebuffer();
        
        // Start playback device after prebuffering
        if (ma_device_start(&playbackDevice) != MA_SUCCESS) {
            printf("[!] Failed to start playback device\n");
            ma_device_stop(&captureDevice);
            return false;
        }
        
        return true;
    }
    
    bool restartPlayback() {
        if (!playbackInitialized) {
            return false;
        }
        
        ma_device_stop(&playbackDevice);
        waitForPrebuffer();
        
        return ma_device_start(&playbackDevice) == MA_SUCCESS;
    }

    void stopDevices() {
        printf("[*] Stopping audio devices...\n");
        
        if (captureInitialized) {
            ma_device_stop(&captureDevice);
        }
        
        if (playbackInitialized) {
            ma_device_stop(&playbackDevice);
        }
        
        // Give callbacks time to exit
        ma_sleep(50);
        printf("[+] Audio devices stopped\n");
    }

    // Forward declarations for callback functions
    static void capture_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount);
    static void playback_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount);
    
private:
    ma_device captureDevice{};
    ma_device playbackDevice{};
    bool captureInitialized = false;
    bool playbackInitialized = false;
    
    void cleanup() {
        if (captureInitialized) {
            ma_device_uninit(&captureDevice);
        }
        if (playbackInitialized) {
            ma_device_uninit(&playbackDevice);
        }
    }
    
    void waitForPrebuffer() {
        while (g_ring.available() < AudioConfig::PREBUFFER_SAMPLES) {
            printf("[*] Waiting for prebuffer... %u samples buffered\r", g_ring.available());
            ma_sleep(AudioConfig::SLEEP_INTERVAL_MS);
        }
        printf("\n[+] Prebuffer complete\n");
    }
};

// === Improved Audio Callbacks ===
void AudioDeviceManager::capture_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount) {
    (void)output; // Explicitly mark as unused
    (void)device; // Explicitly mark as unused

    // Early exit checks - before any processing
    if (!g_running.load(std::memory_order_acquire) || 
        !g_gpuReady.load(std::memory_order_acquire)) {
        return;
    }

    if (!input || !g_upsampler) [[unlikely]] {
        return;
    }

    const auto* inputSamples = static_cast<const float*>(input);
    
    // Cast to VulkanUpsampler to access async API
    auto* vulkanUpsampler = static_cast<VulkanUpsampler*>(g_upsampler.get());
    
    // OPTIMIZED: Check available slots before submission
    // If no slots available, this indicates GPU processing bottleneck
    const size_t availableSlots = vulkanUpsampler->getAvailableSlots();
    
    if (availableSlots == 0) [[unlikely]] {
        // All slots busy - GPU is the bottleneck
        // Track dropped frames for diagnostics
        g_droppedFrames.fetch_add(frameCount, std::memory_order_relaxed);
        
        // Option 1: Drop frames (current behavior - prevents capture thread blocking)
        return;
        
        // Option 2 (alternative): Could wait briefly for a slot, but risks blocking capture thread
        // This is NOT recommended for real-time audio
    }
    
    // Submit work asynchronously (non-blocking)
    // Background polling thread will handle completed work
    uint64_t sequenceId = vulkanUpsampler->processAsync(inputSamples, frameCount, 
        [frameCount](const float* output, uint32_t outputFrames) {
            // Callback executes when GPU work completes (called by polling thread)
            const uint32_t outputSamples = outputFrames * AudioConfig::CHANNELS;
            g_ring.push(output, outputSamples);
            
            // Track processing statistics
            g_processedInputFrames.fetch_add(frameCount, std::memory_order_relaxed);
            g_processedOutputSamples.fetch_add(outputSamples, std::memory_order_relaxed);
        });
    
    // If submission failed despite available slots, there may be other issues
    if (sequenceId == 0 && availableSlots > 0) [[unlikely]] {
        g_droppedFrames.fetch_add(frameCount, std::memory_order_relaxed);
    }
}

void AudioDeviceManager::playback_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount) {
    (void)input;  // Explicitly mark as unused
    (void)device; // Explicitly mark as unused

    // Check if we're shutting down
    if (!g_running.load(std::memory_order_acquire)) {
        auto* outputSamples = static_cast<float*>(output);
        const uint32_t requiredSamples = frameCount * AudioConfig::CHANNELS;
        std::memset(outputSamples, 0, requiredSamples * sizeof(float));
        return;
    }

    auto* outputSamples = static_cast<float*>(output);
    const uint32_t requiredSamples = frameCount * AudioConfig::CHANNELS;

    const uint32_t actualSamples = g_ring.pop(outputSamples, requiredSamples);

    if (actualSamples < requiredSamples) [[unlikely]] {
        // Zero-fill remaining samples
        const size_t remainingBytes = (requiredSamples - actualSamples) * sizeof(float);
        std::memset(outputSamples + actualSamples, 0, remainingBytes);
        
        // Only trigger underrun if buffer is critically low
        const uint32_t remainingBuffer = g_ring.available();
        if (remainingBuffer < AudioConfig::MIN_BUFFER_SAMPLES) {
            g_underrun.store(true, std::memory_order_relaxed);
        }
    }
}

// === Main Processing Loop ===
void runMainLoop(AudioDeviceManager& deviceManager) {
    auto lastStatusTime = std::chrono::steady_clock::now();
    auto startTime = std::chrono::steady_clock::now();
    uint64_t lastDroppedFrames = 0;
    uint64_t lastProcessedInputFrames = 0;
    uint64_t lastProcessedOutputSamples = 0;
    uint32_t lastBufferLevel = 0;
    
    while (g_running) {
        // Non-blocking underrun check
        if (g_underrun.exchange(false)) {
            printf("\n[!] Underrun detected. Re-buffering...\n");

            // Async restart - don't block main loop
            std::thread([&deviceManager]() {
                if (deviceManager.restartPlayback()) {
                    printf("[+] Playback resumed after underrun\n");
                } else {
                    printf("[!] Failed to restart playback after underrun\n");
                }
            }).detach();
        }
        
        // Non-blocking status report (every ~500ms)
        auto now = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatusTime).count();
        
        if (elapsedMs >= 500) {
            const uint32_t bufferLevel = g_ring.available();
            const auto* vulkanUpsampler = static_cast<VulkanUpsampler*>(g_upsampler.get());
            const size_t availableSlots = vulkanUpsampler->getAvailableSlots();
            
            // Calculate throughput metrics
            const uint64_t currentProcessedInput = g_processedInputFrames.load(std::memory_order_relaxed);
            const uint64_t currentProcessedOutput = g_processedOutputSamples.load(std::memory_order_relaxed);
            const uint64_t inputFramesDelta = currentProcessedInput - lastProcessedInputFrames;
            const uint64_t outputSamplesDelta = currentProcessedOutput - lastProcessedOutputSamples;
            
            // Calculate buffer fill rate (samples per second)
            const int32_t bufferDelta = static_cast<int32_t>(bufferLevel) - static_cast<int32_t>(lastBufferLevel);
            const double bufferFillRate = (bufferDelta * 1000.0) / elapsedMs; // samples/sec
            
            // Calculate processing rates
            const double inputRate = (inputFramesDelta * 1000.0) / elapsedMs; // frames/sec
            const double outputRate = (outputSamplesDelta * 1000.0) / elapsedMs; // samples/sec
            
            // Check for dropped frames
            const uint64_t currentDropped = g_droppedFrames.load(std::memory_order_relaxed);
            const uint64_t droppedSinceLastCheck = currentDropped - lastDroppedFrames;
            
            // Calculate total runtime
            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            
            // Status display with comprehensive metrics
            if (droppedSinceLastCheck > 0) {
                printf("[*] Buf: %u (%.1f%%), Fill: %+.0f smp/s, In: %.0f fps, Out: %.0f sps, GPU: %u/%u, DROP: %llu      \r",
                       bufferLevel,
                       (bufferLevel * 100.0f) / AudioConfig::RING_BUFFER_SAMPLES,
                       bufferFillRate,
                       inputRate,
                       outputRate,
                       static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS - availableSlots),
                       static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS),
                       droppedSinceLastCheck * 2); // *2 for per-second rate
            } else {
                printf("[*] Buf: %u (%.1f%%), Fill: %+.0f smp/s, In: %.0f fps, Out: %.0f sps, GPU: %u/%u, Time: %llds    \r",
                       bufferLevel,
                       (bufferLevel * 100.0f) / AudioConfig::RING_BUFFER_SAMPLES,
                       bufferFillRate,
                       inputRate,
                       outputRate,
                       static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS - availableSlots),
                       static_cast<uint32_t>(VulkanUpsampler::NUM_SLOTS),
                       totalElapsed);
            }
            fflush(stdout);
            
            lastDroppedFrames = currentDropped;
            lastProcessedInputFrames = currentProcessedInput;
            lastProcessedOutputSamples = currentProcessedOutput;
            lastBufferLevel = bufferLevel;
            lastStatusTime = now;
        }

        // Very short sleep - let other threads run
        std::this_thread::sleep_for(std::chrono::milliseconds(AudioConfig::MAIN_LOOP_SLEEP_MS));
    }
    printf("\n"); // Clear the status line
    
    // Print final statistics
    auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTime).count();
    const uint64_t totalDropped = g_droppedFrames.load(std::memory_order_relaxed);
    const uint64_t totalProcessed = g_processedInputFrames.load(std::memory_order_relaxed);
    
    if (totalElapsed > 0) {
        printf("[+] Session statistics:\n");
        printf("    Runtime: %lld seconds\n", totalElapsed);
        printf("    Processed: %llu input frames (%.1f fps avg)\n", 
               totalProcessed, totalProcessed / static_cast<double>(totalElapsed));
        printf("    Dropped: %llu frames (%.3f%% loss rate)\n",
               totalDropped, 
               totalProcessed > 0 ? (totalDropped * 100.0 / totalProcessed) : 0.0);
    }
}

// === Main Application ===
int main() {
    std::signal(SIGINT, signal_handler);

    printf("[*] Initializing GPU upsampler...\n");
    
    // Step 1: Initialize GPU upsampler
    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, 
                                AudioConfig::OUTPUT_SAMPLE_RATE, 
                                AudioConfig::CHANNELS)) {
        printf("[!] Failed to initialize upsampler\n");
        return EXIT_FAILURE;
    }

    // Step 2: Load shader
    g_upsampler->setKernel(ResampleKernel::Linear);
    
    // Step 3: Initialize ring buffer
    g_ring.init(AudioConfig::RING_BUFFER_SAMPLES);

    // Step 4: Mark GPU as ready BEFORE initializing audio devices
    // This MUST be done before ma_device_init() calls
    g_gpuReady.store(true, std::memory_order_release);
    
    // Step 5: Start GPU polling thread
    printf("[*] Starting GPU polling thread...\n");
    g_poller.start();
    
    printf("[*] Initializing audio devices...\n");
    
    // Step 6: Initialize audio devices (callbacks may be called during init)
    AudioDeviceManager deviceManager;
    if (!deviceManager.initializeCapture() || !deviceManager.initializePlayback()) {
        g_gpuReady.store(false, std::memory_order_release);
        g_poller.stop();
        return EXIT_FAILURE;
    }

    // Step 7: Start devices
    if (!deviceManager.startDevices()) {
        g_gpuReady.store(false, std::memory_order_release);
        g_poller.stop();
        return EXIT_FAILURE;
    }

    printf("[+] Real-time GPU upsampler started (%u -> %uHz)\n", 
           AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE);
    printf("[*] Press Ctrl+C to stop...\n");

    // Run main processing loop
    runMainLoop(deviceManager);

    // Graceful shutdown sequence
    printf("[*] Initiating shutdown sequence...\n");
    
    // Mark GPU as not ready to stop callbacks
    g_gpuReady.store(false, std::memory_order_release);
    
    // Step 1: Stop GPU polling thread
    printf("[*] Stopping GPU polling thread...\n");
    g_poller.stop();
    
    // Step 2: Stop audio devices to prevent callbacks from running
    deviceManager.stopDevices();
    
    // Step 3: Cleanup GPU resources
    if (g_upsampler) {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    printf("[+] Graceful shutdown complete.\n");
    return EXIT_SUCCESS;
}
