#pragma once
#include "GpuUpsampler.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <array>
#include <queue>
#include <functional>

struct GpuSlot {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    void* inputPtr = nullptr;

    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;
    void* outputPtr = nullptr;

    // Per-slot descriptor set for independent resource binding
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    bool descriptorNeedsUpdate = true;

    // Per-slot buffer size tracking
    VkDeviceSize inputBufferSize = 0;
    VkDeviceSize outputBufferSize = 0;

    // async state
    bool initialized = false;
    bool inFlight = false;
    uint64_t sequenceId = 0;
    uint32_t expectedOutSamples = 0;
    uint32_t skipOffset = 0; // samples to skip due to tail
    
    // Callback storage for async API
    std::function<void(const float*, uint32_t)> callback;
};

/**
 * @brief Vulkan-based GPU audio upsampler implementation.
 */
class VulkanUpsampler : public GpuUpsampler {
public:
    // ZERO-COPY OPTIMIZATION: Callback receives direct pointer to GPU-mapped memory
    // WARNING: Pointer is only valid during callback execution!
    // Data must be consumed or copied within callback before returning.
    using CompletionCallback = std::function<void(const float* output, uint32_t outputFrames)>;

    // === Multi-Slot Configuration ===
    static constexpr uint32_t NUM_SLOTS = 10;  // Increased from 7

    bool initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) override;
    
    void setKernel(ResampleKernel kernel) override;
    
    // === Synchronous API (deprecated - empty stub) ===
    [[deprecated("Use processAsync() for better performance and non-blocking operation")]]
    bool process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) override;
    
    // === Asynchronous API (new) ===
    
    /// @brief Submit work without waiting (fully async)
    /// @param input Input audio data
    /// @param inputFrames Number of input frames
    /// @param callback Called when processing completes (can be nullptr)
    /// @return Sequence ID for tracking, or 0 on failure
    uint64_t processAsync(const float* input, uint32_t inputFrames, CompletionCallback callback = nullptr);
    
    /// @brief Poll and process all completed work without blocking
    /// @return Number of completed works processed
    size_t tryPollAll();
    
    /// @brief Get number of available slots
    size_t getAvailableSlots() const;
    
    // === Low-level API (for manual control) ===
    bool enqueue(const float* input, uint32_t inputFrames);
    bool poll(bool& ready, float* output, uint32_t& outputFrames);
    
    void shutdown() override;

    /// @brief Create shader module from SPIR-V bytecode file
    VkShaderModule createShaderModule(const std::string& filename);

private:
    // === Initialization and Cleanup ===
    
    /// @brief Initialize Vulkan instance, device, and command objects
    bool initVulkan();
    
    /// @brief Clean up all Vulkan resources in proper order
    void cleanupVulkan();
    
    /// @brief Clean up compute pipeline objects
    void cleanupPipeline();

    void cleanupSlotBuffers(GpuSlot& slotRef);

    void cleanupGpuSlot(GpuSlot& cleanupSlot);

    // === Vulkan Initialization Functions ===
    bool createInstance();
    bool selectPhysicalDevice(); 
    bool createLogicalDevice();
    bool createCommandObjects();

    // === Resource Management ===
    
    /// @brief Create input/output buffers for given frame count and slot
    /// @param inputFrames Maximum number of input frames to support
    /// @param slotIndex Target slot index
    bool createBuffers(uint32_t inputFrames, uint32_t slotIndex);
    
    /// @brief Create descriptor set for specific slot
    /// @param slotIndex Target slot index
    bool createDescriptorSet(uint32_t slotIndex);
    
    /// @brief Create compute pipeline with given shader
    /// @param shaderModule Compiled shader module
    bool createPipeline(VkShaderModule shaderModule);

    // === Buffer Management Functions ===
    bool createBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr, const char* label);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    
    // === Data Transfer ===
    
    /// @brief Upload input data to GPU buffer for specific slot
    /// @param input Source audio data
    /// @param totalSamples Number of samples to upload
    /// @param slotIndex Target slot index
    bool uploadInputToGPU(const float* input, uint32_t totalSamples, uint32_t slotIndex);
    
    /// @brief Download processed data from GPU buffer for specific slot
    /// @param output Destination buffer
    /// @param totalSamples Number of samples to download
    /// @param slotIndex Source slot index
    bool downloadOutputFromGPU(float* output, uint32_t totalSamples, uint32_t slotIndex);
    
    /// @brief Dispatch compute shader with given sample counts for specific slot
    /// @param inSamples Input sample count
    /// @param outSamples Output sample count
    /// @param slotIndex Target slot index
    bool dispatch(uint32_t inSamples, uint32_t outSamples, uint32_t slotIndex);

    // === Slot Management ===
    
    /// @brief Find an available slot (not in-flight), actively reclaiming completed work if needed
    /// @return Slot index, or -1 if all slots are busy
    int findAvailableSlot();

    // === Utility Functions ===
    void updateTailBuffer(const float* input, uint32_t inSamples);

    // === Constants ===
    static constexpr uint32_t DEFAULT_CHANNELS = 2;
    
    // OPTIMIZED: Use HOST_VISIBLE + HOST_CACHED for better CPU read performance
    // HOST_CACHED dramatically improves CPU->GPU and GPU->CPU transfer speeds
    static constexpr VkMemoryPropertyFlags BUFFER_MEMORY_PROPS = 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    
    // Fallback without CACHED flag if not available
    static constexpr VkMemoryPropertyFlags BUFFER_MEMORY_PROPS_FALLBACK = 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    /**
     * @brief Push constant structure (must match shader layout exactly)
     */
    struct PushConstants {
        uint32_t inFrameCount;   ///< Number of input frames
        uint32_t outFrameCount;  ///< Number of output frames  
        float ratio;             ///< Resample ratio (output/input)
    };

    // === Configuration ===
    ResampleKernel selectedKernel = ResampleKernel::Linear;
    uint32_t inRate = 0;
    uint32_t outRate = 0;
    uint32_t numChannels = DEFAULT_CHANNELS;
    
    std::vector<float> previousTail;

    // === Memory Management ===
    VkMemoryPropertyFlags inputMemoryProperties = BUFFER_MEMORY_PROPS_FALLBACK;
    VkMemoryPropertyFlags outputMemoryProperties = BUFFER_MEMORY_PROPS_FALLBACK;

    // === Core Vulkan Objects ===
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;

    // === Command Objects ===
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // === Pipeline Objects ===
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // Note: individual descriptor sets now stored per-slot

    // === Multi-Slot State ===
    std::array<GpuSlot, NUM_SLOTS> slots;
    uint32_t currentSlotIndex = 0;  // Round-robin slot selection
    
    // === Async Sequencing ===
    uint64_t nextSequenceId = 0;
    
    // Queue to track submission order for result retrieval
    struct PendingWork {
        uint32_t slotIndex;
        uint64_t sequenceId;
    };
    std::queue<PendingWork> pendingQueue;
};
