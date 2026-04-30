#include "AudioDeviceManager.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace
{
constexpr uint32_t kInputSampleRate = 44100;
constexpr uint32_t kOutputSampleRate = 384000;
constexpr uint32_t kChannels = 2;
} // namespace

AudioDeviceManager::AudioDeviceManager(AudioCallbackContext *callbackContext)
    : callbackContext(callbackContext)
{
}

AudioDeviceManager::~AudioDeviceManager()
{
    cleanup();
}

bool AudioDeviceManager::initializeCapture()
{
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.sampleRate = kInputSampleRate;
    config.capture.format = ma_format_f32;
    config.capture.channels = kChannels;
    config.dataCallback = capture_callback;
    config.pUserData = callbackContext;

    config.noPreSilencedOutputBuffer = MA_TRUE;
    config.noClip = MA_TRUE;
    config.noFixedSizedCallback = MA_TRUE;
    config.performanceProfile = ma_performance_profile_low_latency;

#ifdef MA_WIN32
    config.wasapi.noAutoConvertSRC = MA_TRUE;
    config.wasapi.noDefaultQualitySRC = MA_TRUE;
    config.wasapi.noHardwareOffloading = MA_FALSE;
#endif

    if (ma_device_init(nullptr, &config, &captureDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to initialize capture device\n");
        return false;
    }

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

    printf("[+] Capture device initialized: %u Hz, %u frames/period, %u periods\n", captureDevice.sampleRate,
           captureDevice.capture.internalPeriodSizeInFrames, captureDevice.capture.internalPeriods);

    const float callbackFrequency =
        static_cast<float>(captureDevice.sampleRate) / captureDevice.capture.internalPeriodSizeInFrames;
    const float callbackIntervalMs = 1000.0f / callbackFrequency;

    printf("[+] Capture callback: %.1f Hz (every %.2f ms)\n", callbackFrequency, callbackIntervalMs);

    captureInitialized = true;
    return true;
}

bool AudioDeviceManager::initializePlayback()
{
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.sampleRate = kOutputSampleRate;
    config.playback.format = ma_format_f32;
    config.playback.channels = kChannels;
    config.dataCallback = playback_callback;
    config.pUserData = callbackContext;

    config.noPreSilencedOutputBuffer = MA_TRUE;
    config.noClip = MA_TRUE;
    config.noFixedSizedCallback = MA_TRUE;
    config.performanceProfile = ma_performance_profile_low_latency;

#ifdef MA_WIN32
    config.wasapi.noAutoConvertSRC = MA_TRUE;
    config.wasapi.noDefaultQualitySRC = MA_TRUE;
    config.wasapi.noHardwareOffloading = MA_FALSE;
    config.wasapi.usage = ma_wasapi_usage_pro_audio;
#endif

    if (ma_device_init(nullptr, &config, &playbackDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to initialize playback device\n");
        return false;
    }

    printf("[+] Playback device initialized: %u Hz (requested: %u Hz), %u frames/period, %u periods\n",
           playbackDevice.sampleRate, kOutputSampleRate, playbackDevice.playback.internalPeriodSizeInFrames,
           playbackDevice.playback.internalPeriods);

    playbackInitialized = true;
    return true;
}

bool AudioDeviceManager::startDevices()
{
    if (!captureInitialized || !playbackInitialized)
    {
        return false;
    }

    if (ma_device_start(&captureDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to start capture device\n");
        return false;
    }

    if (ma_device_start(&playbackDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to start playback device\n");
        ma_device_stop(&captureDevice);
        return false;
    }

    return true;
}

bool AudioDeviceManager::startCapture()
{
    if (!captureInitialized)
    {
        return false;
    }

    if (ma_device_start(&captureDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to start capture device\n");
        return false;
    }

    return true;
}

bool AudioDeviceManager::startPlayback()
{
    if (!playbackInitialized)
    {
        return false;
    }

    if (ma_device_start(&playbackDevice) != MA_SUCCESS)
    {
        printf("[!] Failed to start playback device\n");
        return false;
    }

    return true;
}

bool AudioDeviceManager::restartPlayback()
{
    if (!playbackInitialized)
    {
        return false;
    }

    ma_device_stop(&playbackDevice);
    return ma_device_start(&playbackDevice) == MA_SUCCESS;
}

void AudioDeviceManager::stopDevices()
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

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    printf("[+] Audio devices stopped\n");
}

void AudioDeviceManager::cleanup()
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

void AudioDeviceManager::capture_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount)
{
    (void)output;

    auto *context = static_cast<AudioCallbackContext *>(device->pUserData);
    if (!context || !context->running || !context->inputRing || !context->capturedInputFrames)
    {
        return;
    }

    if (!context->running->load(std::memory_order_acquire))
    {
        return;
    }

    if (!input) [[unlikely]]
    {
        return;
    }

    const auto *inputSamples = static_cast<const float *>(input);
    const uint32_t samples = frameCount * context->channels;

    context->inputRing->push(inputSamples, samples);
    context->capturedInputFrames->fetch_add(frameCount, std::memory_order_relaxed);
}

void AudioDeviceManager::playback_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount)
{
    (void)input;

    auto *context = static_cast<AudioCallbackContext *>(device->pUserData);
    if (!context || !context->running || !context->outputRing || !context->playedOutputFrames)
    {
        return;
    }

    auto *outputSamples = static_cast<float *>(output);
    const uint32_t requiredSamples = frameCount * context->channels;

    if (!context->running->load(std::memory_order_acquire))
    {
        std::memset(outputSamples, 0, requiredSamples * sizeof(float));
        return;
    }

    const uint32_t actualSamples = context->outputRing->pop(outputSamples, requiredSamples);
    const uint32_t remainingFrames = context->outputRing->available() / context->channels;

    if (context->minObservedOutputFrames)
    {
        uint32_t currentMin = context->minObservedOutputFrames->load(std::memory_order_relaxed);
        while (remainingFrames < currentMin &&
               !context->minObservedOutputFrames->compare_exchange_weak(currentMin, remainingFrames,
                                                                        std::memory_order_relaxed,
                                                                        std::memory_order_relaxed))
        {
        }
    }

    if (actualSamples < requiredSamples) [[unlikely]]
    {
        const uint32_t zeroFillSamples = requiredSamples - actualSamples;
        const size_t remainingBytes = (requiredSamples - actualSamples) * sizeof(float);
        std::memset(outputSamples + actualSamples, 0, remainingBytes);

        if (context->zeroFillEvents)
        {
            context->zeroFillEvents->fetch_add(1, std::memory_order_relaxed);
        }

        if (context->zeroFillSamples)
        {
            context->zeroFillSamples->fetch_add(zeroFillSamples, std::memory_order_relaxed);
        }

        if (context->underrun && remainingFrames < context->minBufferFrames)
        {
            context->underrun->store(true, std::memory_order_relaxed);
        }
    }

    context->playedOutputFrames->fetch_add(frameCount, std::memory_order_relaxed);
}
