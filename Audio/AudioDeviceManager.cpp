#include "AudioDeviceManager.h"

#include <chrono>
#include <cstdio>
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
