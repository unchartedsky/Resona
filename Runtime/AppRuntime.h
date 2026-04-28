#pragma once

#include "../Audio/AudioCallbackContext.h"
#include "../Audio/AudioDeviceManager.h"

#include <memory>

class AppRuntime
{
  public:
    bool Initialize();
    int Run();
    void Shutdown();

  private:
    AudioCallbackContext audioCallbackContext{};
    std::unique_ptr<AudioDeviceManager> deviceManager;
    bool initialized = false;
};
