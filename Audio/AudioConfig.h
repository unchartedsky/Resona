#pragma once

#include <cstdint>

namespace AudioConfig
{
static constexpr uint32_t INPUT_SAMPLE_RATE = 44100;
static constexpr uint32_t OUTPUT_SAMPLE_RATE = 384000;
static constexpr uint32_t CHANNELS = 2;

static constexpr uint32_t INPUT_RING_BUFFER_FRAMES = 131072;
static constexpr uint32_t OUTPUT_RING_BUFFER_FRAMES = 1048576;

static constexpr uint32_t PREBUFFER_FRAMES = OUTPUT_SAMPLE_RATE * 40 / 100;
static constexpr uint32_t MIN_BUFFER_FRAMES = OUTPUT_SAMPLE_RATE * 25 / 100;

static constexpr uint32_t SLEEP_INTERVAL_MS = 5;
static constexpr uint32_t MAIN_LOOP_SLEEP_MS = 10;
static constexpr uint32_t GPU_THREAD_SLEEP_MS = 1;
} // namespace AudioConfig
