#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "GpuUpsampler.h"
#include "VulkanUpsampler.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
#include <csignal>

std::atomic<bool> g_running{ true };

void signal_handler(int signal)
{
    if (signal == SIGINT) {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_running = false;
    }
}

constexpr uint32_t ringBufferSamples = 384000 * 2; // 1초 (stereo)
constexpr uint32_t maxOutFrames = 48000 * 8;

static std::unique_ptr<GpuUpsampler> g_upsampler;
static float outputBuffer[maxOutFrames * 2]; // Temp buffer for one upsampled batch

// === Ring Buffer ===
struct FloatRingBuffer {
    std::vector<float> buffer;
    std::atomic<uint32_t> writePos{ 0 };
    std::atomic<uint32_t> readPos{ 0 };
    uint32_t size{ 0 };

    void init(uint32_t totalSamples) {
        buffer.resize(totalSamples);
        size = totalSamples;
    }

    void push(const float* data, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            buffer[writePos % size] = data[i];
            writePos++;
        }
    }

    uint32_t pop(float* out, uint32_t count) {
        uint32_t available = writePos.load() - readPos.load();
        uint32_t toRead = (count < available) ? count : available;
        for (uint32_t i = 0; i < toRead; ++i) {
            out[i] = buffer[readPos % size];
            readPos++;
        }
        return toRead;
    }

    uint32_t available() const {
        return writePos.load() - readPos.load();
    }
};

static FloatRingBuffer g_ring;

// === Callbacks ===

void capture_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount)
{
    (void)output; // Unused in capture callback
    (void)device; // Unused in capture callback

    if (!input || !g_upsampler) return;

    const float* in = (const float*)input;
    uint32_t outFrames = 0;

    g_upsampler->process(in, frameCount, outputBuffer, outFrames);
    
    const uint32_t outSamples = outFrames * 2;
    g_ring.push(outputBuffer, outSamples);
}

void playback_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount)
{
    (void)input; // Unused in playback callback
    (void)device; // Unused in playback callback

    float* out = (float*)output;
    const uint32_t requiredSamples = frameCount * 2;

    uint32_t actualSamples = g_ring.pop(out, requiredSamples);

    if (actualSamples < requiredSamples) {
        std::memset(out + actualSamples, 0, (requiredSamples - actualSamples) * sizeof(float));
    }
}

// === Main ===

int main()
{
    std::signal(SIGINT, signal_handler);

    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(44100, 384000, 2)) {
        printf("[!] Failed to initialize upsampler\n");
        return -1;
    }

    g_upsampler->setKernel(ResampleKernel::Linear);
    g_ring.init(ringBufferSamples);

    // --- Capture @ 44100 ---
    ma_device_config capConfig = ma_device_config_init(ma_device_type_capture);
    capConfig.sampleRate = 44100;
    capConfig.capture.format = ma_format_f32;
    capConfig.capture.channels = 2;
    capConfig.dataCallback = capture_callback;

    ma_device captureDevice;
    if (ma_device_init(nullptr, &capConfig, &captureDevice) != MA_SUCCESS) {
        printf("[!] Failed to init capture device\n");
        return -1;
    }

    // --- Playback @ 384000 ---
    ma_device_config playConfig = ma_device_config_init(ma_device_type_playback);
    playConfig.sampleRate = 384000;
    playConfig.playback.format = ma_format_f32;
    playConfig.playback.channels = 2;
    playConfig.dataCallback = playback_callback;

    ma_device playbackDevice;
    if (ma_device_init(nullptr, &playConfig, &playbackDevice) != MA_SUCCESS) {
        printf("[!] Failed to init playback device\n");
        ma_device_uninit(&captureDevice);
        return -1;
    }

    // Start capture device first
    if (ma_device_start(&captureDevice) != MA_SUCCESS) {
        printf("[!] Failed to start capture device\n");
        ma_device_uninit(&playbackDevice);
        return -1;
    }

    // Prebuffer wait
    while (g_ring.available() < 192000) {
        printf("[*] Waiting for prebuffer... %u samples buffered\r", g_ring.available());
        ma_sleep(10);
    }

    // Then start playback
    if (ma_device_start(&playbackDevice) != MA_SUCCESS) {
        printf("[!] Failed to start playback device\n");
        ma_device_uninit(&captureDevice);
        return -1;
    }

    printf("[+] Real-time GPU upsampler started (44100 -> 384000Hz)\n");

    while (g_running) {
        ma_sleep(1000);
    }

    ma_device_uninit(&captureDevice);
    ma_device_uninit(&playbackDevice);
    
    if (g_upsampler) {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    printf("[+] Graceful shutdown complete.\n");
    return 0;
}
