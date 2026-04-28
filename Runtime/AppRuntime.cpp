#include "AppRuntime.h"

#include "../Audio/AudioConfig.h"
#include "RuntimeHooks.h"
#include "../VulkanUpsampler.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

bool AppRuntime::Initialize()
{
    printf("[*] Initializing GPU upsampler...\n");

    g_upsampler = std::make_unique<VulkanUpsampler>();
    if (!g_upsampler->initialize(AudioConfig::INPUT_SAMPLE_RATE, AudioConfig::OUTPUT_SAMPLE_RATE, AudioConfig::CHANNELS))
    {
        printf("[!] Failed to initialize upsampler\n");
        return false;
    }

    g_upsampler->setKernel(ResampleKernel::Nearest);

    printf("[*] Initializing dual ring buffers...\n");
    g_inputRing.init(AudioConfig::INPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);
    g_outputRing.init(AudioConfig::OUTPUT_RING_BUFFER_FRAMES * AudioConfig::CHANNELS);

    g_gpuProcessingContext.gpuReady = &g_gpuReady;
    g_gpuProcessingContext.upsampler = static_cast<VulkanUpsampler *>(g_upsampler.get());
    g_gpuProcessingContext.inputRing = &g_inputRing;
    g_gpuProcessingContext.outputRing = &g_outputRing;
    g_gpuProcessingContext.processedInputFrames = &g_processedInputFrames;
    g_gpuProcessingContext.processedOutputFrames = &g_processedOutputFrames;
    g_gpuProcessingContext.inputSampleRate = AudioConfig::INPUT_SAMPLE_RATE;
    g_gpuProcessingContext.outputSampleRate = AudioConfig::OUTPUT_SAMPLE_RATE;
    g_gpuProcessingContext.channels = AudioConfig::CHANNELS;
    g_gpuProcessingContext.outputRingCapacityFrames = AudioConfig::OUTPUT_RING_BUFFER_FRAMES;
    g_gpuProcessingContext.idleSleepMs = AudioConfig::GPU_THREAD_SLEEP_MS;

    g_gpuReady.store(true, std::memory_order_release);

    printf("[*] Starting GPU processing thread...\n");
    g_gpuProcessor.start();

    printf("[*] Initializing audio devices...\n");
    audioCallbackContext.running = &g_running;
    audioCallbackContext.underrun = &g_underrun;
    audioCallbackContext.inputRing = &g_inputRing;
    audioCallbackContext.outputRing = &g_outputRing;
    audioCallbackContext.capturedInputFrames = &g_capturedInputFrames;
    audioCallbackContext.playedOutputFrames = &g_playedOutputFrames;
    audioCallbackContext.channels = AudioConfig::CHANNELS;
    audioCallbackContext.minBufferFrames = AudioConfig::MIN_BUFFER_FRAMES;

    deviceManager = std::make_unique<AudioDeviceManager>(&audioCallbackContext);
    if (!deviceManager->initializeCapture() || !deviceManager->initializePlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager.reset();
        return false;
    }

    if (!deviceManager->startCapture())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager.reset();
        return false;
    }

    waitForOutputPrebuffer();

    if (!deviceManager->startPlayback())
    {
        g_gpuReady.store(false, std::memory_order_release);
        g_gpuProcessor.stop();
        deviceManager->stopDevices();
        deviceManager.reset();
        return false;
    }

    if (g_upsampler)
    {
        static_cast<VulkanUpsampler *>(g_upsampler.get())->resetAdaptiveTarget();
    }

    printf("[+] Real-time GPU upsampler started (%u -> %uHz)\n", AudioConfig::INPUT_SAMPLE_RATE,
           AudioConfig::OUTPUT_SAMPLE_RATE);
    printf("[*] Architecture: Capture -> Input Buffer (44.1kHz) -> GPU -> Output Buffer (384kHz) -> Playback\n");
    printf("[*] Press Ctrl+C to stop...\n");

    initialized = true;
    return true;
}

int AppRuntime::Run()
{
    if (!initialized || !deviceManager)
    {
        return EXIT_FAILURE;
    }

    runMainLoop(*deviceManager);
    return EXIT_SUCCESS;
}

void AppRuntime::Shutdown()
{
    if (!initialized)
    {
        return;
    }

    printf("[*] Initiating shutdown sequence...\n");

    g_running.store(false, std::memory_order_release);
    g_gpuReady.store(false, std::memory_order_release);

    printf("[*] Stopping GPU processing thread...\n");
    g_gpuProcessor.stop();

    if (deviceManager)
    {
        deviceManager->stopDevices();
        deviceManager.reset();
    }

    if (g_upsampler)
    {
        g_upsampler->shutdown();
        g_upsampler.reset();
    }

    initialized = false;
    printf("[+] Graceful shutdown complete.\n");
}
