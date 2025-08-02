#pragma once
#include "GpuUpsampler.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

struct GpuSlot {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    void* inputPtr = nullptr;

    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;
    void* outputPtr = nullptr;

    bool initialized = false;
};

/**
 * @brief Vulkan-based GPU audio upsampler implementation.
 */
class VulkanUpsampler : public GpuUpsampler {
public:
    bool initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) override;
    void setKernel(ResampleKernel kernel) override;
    bool process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) override;
    void shutdown() override;

    /// @brief Create shader module from SPIR-V bytecode file
    /// @param filename Path to compiled shader file
    /// @return Valid shader module or VK_NULL_HANDLE on failure
    VkShaderModule createShaderModule(const std::string& filename);

private:
    // add state tracking for optimization
    bool descriptorSetNeedsUpdate = true;
    bool buffersChanged = false;
    VkDeviceSize lastInputBufferSize = 0;
    VkDeviceSize lastOutputBufferSize = 0;

    // === Initialization and Cleanup ===
    
    /// @brief Initialize Vulkan instance, device, and command objects
    bool initVulkan();
    
    /// @brief Clean up all Vulkan resources in proper order
    void cleanupVulkan();
    
    /// @brief Clean up compute pipeline objects
    void cleanupPipeline();
    
    /// @brief Clean up buffer and memory objects  
    void cleanupBuffers();

    // === Vulkan Initialization Functions ===
    bool createInstance();
    bool selectPhysicalDevice(); 
    bool createLogicalDevice();
    bool createCommandObjects();

    // === Resource Management ===
    
    /// @brief Create input/output buffers for given frame count
    /// @param inputFrames Maximum number of input frames to support
    bool createBuffers(uint32_t inputFrames);
    
    /// @brief Create and bind descriptor sets for buffers
    bool createDescriptorSet();
    
    /// @brief Create compute pipeline with given shader
    /// @param shaderModule Compiled shader module
    bool createPipeline(VkShaderModule shaderModule);

    // === Buffer Management Functions ===
    bool createBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr, const char* label);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    
    // === Data Transfer ===
    
    /// @brief Upload input data to GPU buffer
    /// @param input Source audio data
    /// @param totalSamples Number of samples to upload
    bool uploadInputToGPU(const float* input, uint32_t totalSamples);
    
    /// @brief Download processed data from GPU buffer
    /// @param output Destination buffer
    /// @param totalSamples Number of samples to download
    bool downloadOutputFromGPU(float* output, uint32_t totalSamples);
    
    /// @brief Dispatch compute shader with given sample counts
    /// @param inSamples Input sample count
    /// @param outSamples Output sample count  
    bool dispatch(uint32_t inSamples, uint32_t outSamples);

    // === Utility Functions ===
    void updateTailBuffer(const float* input, uint32_t inSamples);

    // === Constants ===
    static constexpr uint32_t DEFAULT_CHANNELS = 2;
    static constexpr VkMemoryPropertyFlags BUFFER_MEMORY_PROPS = 
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
    uint32_t maxInputFrames = 0;
    uint32_t maxOutputSamples = 0;
    std::vector<float> previousTail;

    // === Performance Buffers ===
    std::vector<float> workingInputBuffer;   ///< Reusable buffer for input processing
    std::vector<float> workingOutputBuffer;  ///< Reusable buffer for output processing

    // === Memory Management ===
    void* inputMappedPtr = nullptr;
    void* outputMappedPtr = nullptr;
    VkMemoryPropertyFlags inputMemoryProperties = BUFFER_MEMORY_PROPS;
    VkMemoryPropertyFlags outputMemoryProperties = BUFFER_MEMORY_PROPS;

    // === Core Vulkan Objects ===
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;

    // === Command Objects ===
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // === Buffer Objects ===
    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;

    // === Pipeline Objects ===
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkFence persistentFence = VK_NULL_HANDLE;  ///< Reusable fence for GPU synchronization

    GpuSlot slot;
};
