#pragma once
#include "GpuUpsampler.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

/**
 * @brief Vulkan-based GPU audio upsampler implementation.
 */
class VulkanUpsampler : public GpuUpsampler {
public:
    bool initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) override;
    void setKernel(ResampleKernel kernel) override;
    bool process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) override;
    void shutdown() override;

    VkShaderModule createShaderModule(const std::string& filename);

private:
    bool initVulkan();
    void cleanupVulkan();
    void cleanupPipeline();
    void cleanupBuffers();

    bool createBuffers(uint32_t inputFrames);
    bool createDescriptorSet();
    bool createPipeline(VkShaderModule shaderModule);
    bool uploadInputToGPU(const float* input, uint32_t totalSamples);
    bool downloadOutputFromGPU(float* output, uint32_t totalSamples);
    bool dispatch(uint32_t inSamples, uint32_t outSamples);

    /**
     * @brief Push constant structure (must match shader layout)
     */
    struct PushConstants {
        uint32_t inFrameCount;
        uint32_t outFrameCount;
        float ratio;
    };

    std::vector<float> previousTail;

    // === Configuration ===
    ResampleKernel selectedKernel = ResampleKernel::Linear;
    uint32_t inRate = 0;
    uint32_t outRate = 0;
    uint32_t numChannels = 2;
    uint32_t maxInputFrames = 0;

    // === Vulkan handles ===
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;

    VkShaderModule shaderModule = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};
