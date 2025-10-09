#define NOMINMAX

#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
#include <csignal>
#include <cstdlib>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "GpuUpsampler.h"
#include "VulkanUpsampler.h"

// === Configuration Constants ===
namespace AudioConfig {
    static constexpr uint32_t INPUT_SAMPLE_RATE = 44100;
    static constexpr uint32_t OUTPUT_SAMPLE_RATE = 384000;
    static constexpr uint32_t CHANNELS = 2;
    static constexpr uint32_t PREBUFFER_SAMPLES = OUTPUT_SAMPLE_RATE / 2; // 0.5 seconds
    static constexpr uint32_t SLEEP_INTERVAL_MS = 10;
    static constexpr uint32_t MAIN_LOOP_SLEEP_MS = 100;
    
    // Use power-of-2 size for optimal ring buffer performance
    // 2^21 = 2097152 samples ~= 2.73 seconds at 384kHz (was 1536000 = 4 seconds)
    static constexpr uint32_t RING_BUFFER_SAMPLES = 2097152;
}

// === Global State ===
std::atomic<bool> g_running{ true };
std::atomic<bool> g_underrun{ false };
std::atomic<bool> g_gpuReady{ false };

static std::unique_ptr<GpuUpsampler> g_upsampler;

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
    }

    void push(const float* data, uint32_t count) noexcept {
        if (count > size) [[unlikely]] {
            printf("[!] Push count exceeds buffer size\n");
            return;
        }
        
        const uint32_t writeIdx = writePos.load(std::memory_order_relaxed);
        
        // Check if we can copy to contiguous memory region
        const uint32_t endSpace = size - (writeIdx % size);
        if (count <= endSpace) {
            // Single contiguous copy
            std::memcpy(&buffer[writeIdx % size], data, count * sizeof(float));
        } else {
            // Split copy across buffer boundary
            std::memcpy(&buffer[writeIdx % size], data, endSpace * sizeof(float));
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

        // Check if we can copy from contiguous memory region
        const uint32_t endSpace = size - (readIdx % size);
        if (toRead <= endSpace) {
            // Single contiguous copy
            std::memcpy(out, &buffer[readIdx % size], toRead * sizeof(float));
        } else {
            // Split copy across buffer boundary
            std::memcpy(out, &buffer[readIdx % size], endSpace * sizeof(float));
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

        if (ma_device_init(nullptr, &config, &captureDevice) != MA_SUCCESS) {
            printf("[!] Failed to initialize capture device\n");
            return false;
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
        config.noPreSilencedOutputBuffer = MA_TRUE;  // Prevent pre-initialization testing

        if (ma_device_init(nullptr, &config, &playbackDevice) != MA_SUCCESS) {
            printf("[!] Failed to initialize playback device\n");
            return false;
        }
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
    
    // Submit work asynchronously (non-blocking)
    uint64_t sequenceId = vulkanUpsampler->processAsync(inputSamples, frameCount, 
        [](const float* output, uint32_t outputFrames) {
            // Callback executes when GPU work completes
            const uint32_t outputSamples = outputFrames * AudioConfig::CHANNELS;
            g_ring.push(output, outputSamples);
        });
    
    if (sequenceId == 0) [[unlikely]] {
        // Reduce log spam - only log periodically (skip early failures during init)
        static thread_local uint32_t errorCount = 0;
        if (++errorCount >= 10 && errorCount % 100 == 10) {  // Skip first 10 errors, then log every 100th
            printf("[!] GPU submission failed (all slots busy) - count: %u\n", errorCount);
        }
    }
    
    // Poll completed work (non-blocking) - callbacks will be invoked
    static thread_local std::vector<AsyncResult> results;
    results.clear();
    vulkanUpsampler->tryPollAll(results);
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
        g_underrun.store(true, std::memory_order_relaxed);
    }
}

// === Main Processing Loop ===
void runMainLoop(AudioDeviceManager& deviceManager) {
    while (g_running) {
        if (g_underrun.exchange(false)) {
            printf("\n[!] Underrun detected. Re-buffering...\n");

            if (deviceManager.restartPlayback()) {
                printf("[+] Playback resumed after underrun\n");
            } else {
                printf("[!] Failed to restart playback after underrun\n");
                break;
            }
        }

        ma_sleep(AudioConfig::MAIN_LOOP_SLEEP_MS);
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
    
    printf("[*] Initializing audio devices...\n");
    
    // Step 5: Initialize audio devices (callbacks may be called during init)
    AudioDeviceManager deviceManager;
    if (!deviceManager.initializeCapture() || !deviceManager.initializePlayback()) {
        g_gpuReady.store(false, std::memory_order_release);
        return EXIT_FAILURE;
    }

    // Step 6: Start devices
    if (!deviceManager.startDevices()) {
        g_gpuReady.store(false, std::memory_order_release);
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
    
    // Step 1: Stop audio devices first to prevent callbacks from running
    deviceManager.stopDevices();
    
    // Step 2: Cleanup GPU resources
    if (g_upsampler) {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    printf("[+] Graceful shutdown complete.\n");
    return EXIT_SUCCESS;
}
