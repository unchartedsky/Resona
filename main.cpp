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
    static constexpr uint32_t BUFFER_MULTIPLIER = 4;
    static constexpr uint32_t MAX_FRAME_MULTIPLIER = 8;
    static constexpr uint32_t PREBUFFER_SAMPLES = OUTPUT_SAMPLE_RATE / 2; // 0.5 seconds
    static constexpr uint32_t SLEEP_INTERVAL_MS = 10;
    static constexpr uint32_t MAIN_LOOP_SLEEP_MS = 100;
    
    static constexpr uint32_t RING_BUFFER_SAMPLES = OUTPUT_SAMPLE_RATE * BUFFER_MULTIPLIER;
    static constexpr uint32_t MAX_OUTPUT_FRAMES = INPUT_SAMPLE_RATE * MAX_FRAME_MULTIPLIER;
}

// === Global State ===
std::atomic<bool> g_running{ true };
std::atomic<bool> g_underrun{ false };

static std::unique_ptr<GpuUpsampler> g_upsampler;
static float outputBuffer[AudioConfig::MAX_OUTPUT_FRAMES * AudioConfig::CHANNELS];

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

    if (!input || !g_upsampler) [[unlikely]] {
        return;
    }

    const auto* inputSamples = static_cast<const float*>(input);
    uint32_t outputFrames = 0;

    if (g_upsampler->process(inputSamples, frameCount, outputBuffer, outputFrames)) {
        const uint32_t outputSamples = outputFrames * AudioConfig::CHANNELS;
        g_ring.push(outputBuffer, outputSamples);
    } else [[unlikely]] {
        printf("[!] GPU processing failed\n");
    }
}

void AudioDeviceManager::playback_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount) {
    (void)input;  // Explicitly mark as unused
    (void)device; // Explicitly mark as unused

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

    // Initialize GPU upsampler
    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, 
                                AudioConfig::OUTPUT_SAMPLE_RATE, 
                                AudioConfig::CHANNELS)) {
        printf("[!] Failed to initialize upsampler\n");
        return EXIT_FAILURE;
    }

    g_upsampler->setKernel(ResampleKernel::Linear);
    g_ring.init(AudioConfig::RING_BUFFER_SAMPLES);

    // Initialize and start audio devices
    AudioDeviceManager deviceManager;
    if (!deviceManager.initializeCapture() || !deviceManager.initializePlayback()) {
        return EXIT_FAILURE;
    }

    if (!deviceManager.startDevices()) {
        return EXIT_FAILURE;
    }

    printf("[+] Real-time GPU upsampler started (%u -> %uHz)\n", 
           AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE);

    // Run main processing loop
    runMainLoop(deviceManager);

    // Cleanup resources
    if (g_upsampler) {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    printf("[+] Graceful shutdown complete.\n");
    return EXIT_SUCCESS;
}
