#pragma once

#include "../Audio/AudioCallbackContext.h"
#include "../Audio/AudioDeviceManager.h"

#include <chrono>

#include <memory>

class AppRuntime
{
  public:
    bool Initialize();
    int Run();
    void Shutdown();

  private:
    void waitForOutputPrebuffer() const;
    void runMainLoop();

    AudioCallbackContext audioCallbackContext{};
    std::unique_ptr<AudioDeviceManager> deviceManager;
    bool initialized = false;
};
