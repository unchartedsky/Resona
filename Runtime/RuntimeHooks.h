#pragma once

#include "../Audio/FloatRingBuffer.h"
#include "../GpuUpsampler.h"
#include "../RenderPipeline/GpuProcessingThread.h"

#include <atomic>
#include <cstdint>
#include <memory>

extern std::atomic<bool> g_running;
extern std::atomic<bool> g_underrun;
extern std::atomic<bool> g_restartRequested;
extern std::atomic<bool> g_restartInProgress;
extern std::atomic<bool> g_gpuReady;
extern std::atomic<uint64_t> g_capturedInputFrames;
extern std::atomic<uint64_t> g_processedInputFrames;
extern std::atomic<uint64_t> g_processedOutputFrames;
extern std::atomic<uint64_t> g_playedOutputFrames;

extern std::unique_ptr<GpuUpsampler> g_upsampler;
extern FloatRingBuffer g_inputRing;
extern FloatRingBuffer g_outputRing;
extern GpuProcessingContext g_gpuProcessingContext;
extern GpuProcessingThread g_gpuProcessor;

void waitForOutputPrebuffer();
void runMainLoop(class AudioDeviceManager &deviceManager);
