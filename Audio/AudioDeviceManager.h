#pragma once

#include "AudioCallbackContext.h"
#include "../miniaudio.h"

class AudioDeviceManager
{
  public:
    explicit AudioDeviceManager(AudioCallbackContext *callbackContext);
    ~AudioDeviceManager();

    bool initializeCapture();
    bool initializePlayback();
    bool startDevices();
    bool startCapture();
    bool startPlayback();
    bool restartPlayback();
    void stopDevices();

    static void capture_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount);
    static void playback_callback(ma_device *device, void *output, const void *input, ma_uint32 frameCount);

  private:
    AudioCallbackContext *callbackContext = nullptr;
    ma_device captureDevice{};
    ma_device playbackDevice{};
    bool captureInitialized = false;
    bool playbackInitialized = false;

    void cleanup();
};
