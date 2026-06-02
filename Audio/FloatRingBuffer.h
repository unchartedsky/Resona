#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Compiler-specific prefetch intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#define RESONA_PREFETCH_READ(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#define RESONA_PREFETCH_WRITE(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define RESONA_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#define RESONA_PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
#define RESONA_PREFETCH_READ(addr) ((void)0)
#define RESONA_PREFETCH_WRITE(addr) ((void)0)
#endif

class FloatRingBuffer
{
  public:
    void init(uint32_t totalSamples)
    {
        buffer.resize(totalSamples);
        size = totalSamples;
        sizeMask = 0;
        useFastModulo = false;
        writePos.store(0, std::memory_order_relaxed);
        readPos.store(0, std::memory_order_relaxed);

        const bool isPowerOf2 = (size & (size - 1)) == 0;
        if (!isPowerOf2)
        {
            printf("[!] Warning: Ring buffer size is not power of 2, performance may be suboptimal\n");
        }
        else
        {
            sizeMask = size - 1;
            useFastModulo = true;
            printf("[+] Ring buffer using fast power-of-2 modulo optimization\n");
        }

        printf("[+] Ring buffer initialized: %u samples (%.1f KB)\n", size, (size * sizeof(float)) / 1024.0f);
    }

    void push(const float *data, uint32_t count) noexcept
    {
        if (count > size) [[unlikely]]
        {
            printf("[!] Push count exceeds buffer size\n");
            return;
        }

        const uint32_t writeIdx = writePos.load(std::memory_order_relaxed);
        const uint32_t writeOffset = useFastModulo ? (writeIdx & sizeMask) : (writeIdx % size);

        if (count > 64)
        {
            RESONA_PREFETCH_WRITE(&buffer[writeOffset]);
        }

        const uint32_t endSpace = size - writeOffset;
        if (count <= endSpace)
        {
            std::memcpy(&buffer[writeOffset], data, count * sizeof(float));
        }
        else
        {
            std::memcpy(&buffer[writeOffset], data, endSpace * sizeof(float));
            std::memcpy(&buffer[0], data + endSpace, (count - endSpace) * sizeof(float));
        }

        writePos.store(writeIdx + count, std::memory_order_release);
    }

    uint32_t pop(float *out, uint32_t count) noexcept
    {
        const uint32_t readIdx = readPos.load(std::memory_order_relaxed);
        const uint32_t writeIdx = writePos.load(std::memory_order_acquire);
        const uint32_t available = writeIdx - readIdx;
        const uint32_t toRead = std::min(count, available);

        if (toRead == 0) [[unlikely]]
        {
            return 0;
        }

        const uint32_t readOffset = useFastModulo ? (readIdx & sizeMask) : (readIdx % size);

        if (toRead > 64)
        {
            RESONA_PREFETCH_READ(&buffer[readOffset]);
        }

        const uint32_t endSpace = size - readOffset;
        if (toRead <= endSpace)
        {
            std::memcpy(out, &buffer[readOffset], toRead * sizeof(float));
        }
        else
        {
            std::memcpy(out, &buffer[readOffset], endSpace * sizeof(float));
            std::memcpy(out + endSpace, &buffer[0], (toRead - endSpace) * sizeof(float));
        }

        readPos.store(readIdx + toRead, std::memory_order_release);
        return toRead;
    }

    uint32_t available() const noexcept
    {
        return writePos.load(std::memory_order_acquire) - readPos.load(std::memory_order_relaxed);
    }

  private:
    std::vector<float> buffer;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
    alignas(64) std::atomic<uint32_t> writePos{0};
    alignas(64) std::atomic<uint32_t> readPos{0};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    uint32_t size{0};
    uint32_t sizeMask{0};
    bool useFastModulo{false};
};
