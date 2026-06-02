#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

class OutputBufferQueue
{
  public:
    using ReleaseCallback = void (*)(void *context, uint32_t token);

    void init(uint32_t totalSamples, uint32_t maxBlocks)
    {
        blocks.clear();
        blocks.resize(maxBlocks + 1);
        capacitySamples = totalSamples;
        writeBlock.store(0, std::memory_order_relaxed);
        readBlock.store(0, std::memory_order_relaxed);
        availableSamples.store(0, std::memory_order_relaxed);
    }

    bool pushBlock(const float *data, uint32_t count, void *ownerContext, uint32_t ownerToken,
                   ReleaseCallback releaseCallback) noexcept
    {
        if (count == 0)
        {
            if (releaseCallback)
            {
                releaseCallback(ownerContext, ownerToken);
            }
            return true;
        }

        if (blocks.size() < 2 || count > capacitySamples)
        {
            return false;
        }

        const uint32_t currentAvailable = availableSamples.load(std::memory_order_acquire);
        if (currentAvailable + count > capacitySamples)
        {
            return false;
        }

        const uint32_t writeIndex = writeBlock.load(std::memory_order_relaxed);
        const uint32_t nextWriteIndex = advance(writeIndex);
        if (nextWriteIndex == readBlock.load(std::memory_order_acquire))
        {
            return false;
        }

        Block &block = blocks[writeIndex];
        block.data = data;
        block.sampleCount = count;
        block.consumedSamples = 0;
        block.ownerContext = ownerContext;
        block.ownerToken = ownerToken;
        block.releaseCallback = releaseCallback;

        writeBlock.store(nextWriteIndex, std::memory_order_release);
        availableSamples.fetch_add(count, std::memory_order_release);
        return true;
    }

    uint32_t pop(float *out, uint32_t count) noexcept
    {
        uint32_t copiedSamples = 0;
        uint32_t readIndex = readBlock.load(std::memory_order_relaxed);
        const uint32_t writeIndex = writeBlock.load(std::memory_order_acquire);

        while (copiedSamples < count && readIndex != writeIndex)
        {
            Block &block = blocks[readIndex];
            const uint32_t remainingSamples = block.sampleCount - block.consumedSamples;
            const uint32_t samplesToCopy = std::min(count - copiedSamples, remainingSamples);

            std::memcpy(out + copiedSamples, block.data + block.consumedSamples, samplesToCopy * sizeof(float));
            copiedSamples += samplesToCopy;
            block.consumedSamples += samplesToCopy;

            if (block.consumedSamples == block.sampleCount)
            {
                if (block.releaseCallback)
                {
                    block.releaseCallback(block.ownerContext, block.ownerToken);
                }

                block = {};
                readIndex = advance(readIndex);
                readBlock.store(readIndex, std::memory_order_release);
            }
        }

        if (copiedSamples > 0)
        {
            availableSamples.fetch_sub(copiedSamples, std::memory_order_release);
        }

        return copiedSamples;
    }

    uint32_t available() const noexcept
    {
        return availableSamples.load(std::memory_order_acquire);
    }

    void reset() noexcept
    {
        if (blocks.empty())
        {
            return;
        }

        uint32_t readIndex = readBlock.load(std::memory_order_relaxed);
        const uint32_t writeIndex = writeBlock.load(std::memory_order_relaxed);
        while (readIndex != writeIndex)
        {
            Block &block = blocks[readIndex];
            if (block.releaseCallback)
            {
                block.releaseCallback(block.ownerContext, block.ownerToken);
            }

            block = {};
            readIndex = advance(readIndex);
        }

        readBlock.store(0, std::memory_order_relaxed);
        writeBlock.store(0, std::memory_order_relaxed);
        availableSamples.store(0, std::memory_order_relaxed);
    }

  private:
    struct Block
    {
        const float *data = nullptr;
        uint32_t sampleCount = 0;
        uint32_t consumedSamples = 0;
        void *ownerContext = nullptr;
        uint32_t ownerToken = 0;
        ReleaseCallback releaseCallback = nullptr;
    };

    uint32_t advance(uint32_t index) const noexcept
    {
        return static_cast<uint32_t>((index + 1) % blocks.size());
    }

    std::vector<Block> blocks;

    alignas(64) std::atomic<uint32_t> writeBlock{0};
    alignas(64) std::atomic<uint32_t> readBlock{0};
    alignas(64) std::atomic<uint32_t> availableSamples{0};

    uint32_t capacitySamples{0};
}
;