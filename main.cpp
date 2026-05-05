#define NOMINMAX

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "Audio/FloatRingBuffer.h"
#include "GpuUpsampler.h"
#include "RenderPipeline/GpuProcessingThread.h"
#include "Runtime/AppRuntime.h"

// === Global State ===
std::atomic<bool> g_running{true};
std::atomic<bool> g_underrun{false};
std::atomic<bool> g_restartRequested{false};
std::atomic<bool> g_restartInProgress{false};
std::atomic<bool> g_gpuReady{false};
std::atomic<uint64_t> g_capturedInputFrames{0};   // Track total captured input frames
std::atomic<uint64_t> g_processedInputFrames{0};  // Track total processed input frames
std::atomic<uint64_t> g_processedOutputFrames{0}; // Track total output frames added to output buffer
std::atomic<uint64_t> g_playedOutputFrames{0};    // Track total output frames played
std::atomic<uint64_t> g_zeroFillEvents{0};        // Track playback callbacks that had to zero-fill
std::atomic<uint64_t> g_zeroFillSamples{0};       // Track total zero-filled samples
std::atomic<uint32_t> g_minObservedOutputFrames{UINT32_MAX}; // Track worst-case output buffer depth

std::unique_ptr<GpuUpsampler> g_upsampler;

// === Dual Ring Buffers ===
FloatRingBuffer g_inputRing;  // 44.1kHz captured audio
FloatRingBuffer g_outputRing; // 384kHz upsampled audio

GpuProcessingContext g_gpuProcessingContext;
GpuProcessingThread g_gpuProcessor(&g_gpuProcessingContext);

// === Signal Handler ===
void signal_handler(int signal)
{
    if (signal == SIGINT)
    {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_running = false;
    }
}

int main()
{
    std::signal(SIGINT, signal_handler);

    AppRuntime appRuntime;
    if (!appRuntime.Initialize())
    {
        return EXIT_FAILURE;
    }

    const int exitCode = appRuntime.Run();
    appRuntime.Shutdown();
    return exitCode;
}
