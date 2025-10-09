#include "VulkanUpsampler.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <thread>

// === Configuration Constants ===
namespace VulkanConfig {
    static constexpr uint32_t WORKGROUP_SIZE = 64;
    static constexpr uint32_t MAX_TAIL_FRAMES = 4;
    static constexpr uint32_t API_VERSION = VK_API_VERSION_1_2;
    static constexpr float QUEUE_PRIORITY = 1.0f;
    static constexpr const char* APP_NAME = "VulkanUpsampler";
    static constexpr const char* ENGINE_NAME = "No Engine";
    static constexpr uint64_t FENCE_TIMEOUT = UINT64_MAX;
}

// === Public Interface Implementation ===

bool VulkanUpsampler::initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) {
    inRate = inputRate;
    outRate = outputRate;
    numChannels = channels;

    if (!initVulkan()) {
        printf("[!] Vulkan initialization failed\n");
        return false;
    }

    // Create fence for async synchronization
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS) {
        printf("[!] Failed to create slot fence\n");
        return false;
    }

    // Allocate initial GPU buffers
    VkDeviceSize inputSize = 4096 * sizeof(float);
    if (!createBuffer(inputSize, slot.inputBuffer, slot.inputMemory, &slot.inputPtr, "slot.input")) {
        return false;
    }

    VkDeviceSize outputSize = 8192 * sizeof(float);
    if (!createBuffer(outputSize, slot.outputBuffer, slot.outputMemory, &slot.outputPtr, "slot.output")) {
        return false;
    }

    slot.initialized = true;
    lastInputBufferSize = inputSize;
    lastOutputBufferSize = outputSize;
    descriptorSetNeedsUpdate = true;

    // Pre-allocate working buffers
    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t estimatedInputSamples = 2048 * numChannels;
    const uint32_t estimatedOutputSamples = static_cast<uint32_t>(estimatedInputSamples * ratio);
    
    workingInputBuffer.reserve(estimatedInputSamples * 2);
    workingOutputBuffer.reserve(estimatedOutputSamples * 2);
    
    printf("[+] VulkanUpsampler initialized: %uHz -> %uHz (%u channels)\n", 
           inputRate, outputRate, channels);

    return true;
}

void VulkanUpsampler::setKernel(ResampleKernel kernel) {
    selectedKernel = kernel;

    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }

    std::string filename;
    switch (kernel) {
    case ResampleKernel::Linear:
        filename = "linear.spv";
        break;
    case ResampleKernel::Cubic:
        filename = "cubic.spv";
        break;
    case ResampleKernel::Sinc:
        filename = "sinc.spv";
        break;
    default:
        printf("[!] Unknown resampling kernel selected\n");
        return;
    }

    shaderModule = createShaderModule(filename);
    if (shaderModule == VK_NULL_HANDLE) {
        printf("[!] Failed to load shader module: %s\n", filename.c_str());
    }
}

bool VulkanUpsampler::process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) {
    if (!enqueue(input, inputFrames)) {
        return false;
    }
    
    bool ready = false;
    // Poll for completion (max 1000 iterations for real-time constraints)
    // With 1ms timeout per poll, worst case = 1 second
    for (int i = 0; i < 1000 && !ready; ++i) {
        if (!poll(ready, output, outputFrames)) {
            return false;
        }
        
        // Early exit if ready
        if (ready) {
            return true;
        }
    }
    
    if (!ready) {
        printf("[!] GPU processing timeout\n");
        return false;
    }
    
    return true;
}

void VulkanUpsampler::shutdown() {
    // Wait for any in-flight GPU work to complete before cleanup
    if (slot.inFlight && slot.fence != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &slot.fence, VK_TRUE, VulkanConfig::FENCE_TIMEOUT);
        slot.inFlight = false;
    }

    if (slot.initialized) {
        cleanupGpuSlot(slot);
    }

    cleanupPipeline();
    cleanupVulkan();
    
    printf("[+] VulkanUpsampler shutdown complete\n");
}

// === Private Implementation ===

bool VulkanUpsampler::initVulkan() {
    if (!createInstance()) return false;
    if (!selectPhysicalDevice()) return false;
    if (!createLogicalDevice()) return false;
    if (!createCommandObjects()) return false;
    
    printf("[+] Vulkan initialization successful\n");
    return true;
}

bool VulkanUpsampler::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = VulkanConfig::APP_NAME;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = VulkanConfig::ENGINE_NAME;
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VulkanConfig::API_VERSION;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        printf("[!] Failed to create Vulkan instance\n");
        return false;
    }

    return true;
}

bool VulkanUpsampler::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        printf("[!] No Vulkan-compatible GPUs found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Select first suitable device (could be improved with device scoring)
    physicalDevice = devices[0];
    
    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    printf("[+] Selected GPU: %s\n", deviceProps.deviceName);
    
    return true;
}

bool VulkanUpsampler::createLogicalDevice() {
    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int computeQueueFamily = -1;
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }

    if (computeQueueFamily == -1) {
        printf("[!] No compute-capable queue family found\n");
        return false;
    }

    // Create logical device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &VulkanConfig::QUEUE_PRIORITY;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        printf("[!] Failed to create logical device\n");
        return false;
    }

    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    return true;
}

bool VulkanUpsampler::createCommandObjects() {
    // Get queue family index for command pool
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t computeQueueFamily = 0;
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = i;
            break;
        }
    }

    // Create command pool with proper flags for command buffer reset
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    // CRITICAL FIX: Need RESET_COMMAND_BUFFER_BIT to allow individual command buffer reset
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        printf("[!] Failed to create command pool\n");
        return false;
    }

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &slot.commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to allocate command buffer\n");
        return false;
    }

    return true;
}

VkShaderModule VulkanUpsampler::createShaderModule(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        printf("[!] Failed to open shader file: %s\n", filename.c_str());
        return VK_NULL_HANDLE;
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule module;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        printf("[!] Failed to create shader module\n");
        return VK_NULL_HANDLE;
    }

    printf("[+] Shader loaded: %s (%.1f KB)\n", filename.c_str(), fileSize / 1024.0);
    return module;
}

bool VulkanUpsampler::createBuffers(uint32_t inputFrames) {
    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t totalInputSamples = inputFrames * numChannels;
    const uint32_t totalOutputSamples = static_cast<uint32_t>(inputFrames * ratio) * numChannels;

    const VkDeviceSize requiredInputSize = sizeof(float) * totalInputSamples;
    const VkDeviceSize requiredOutputSize = sizeof(float) * totalOutputSamples;

    // Check if resize needed
    bool needsResize = false;
    if (requiredInputSize > lastInputBufferSize) {
        lastInputBufferSize = requiredInputSize;
        needsResize = true;
    }
    if (requiredOutputSize > lastOutputBufferSize) {
        lastOutputBufferSize = requiredOutputSize;
        needsResize = true;
    }

    if (!needsResize) {
        return true;
    }

    cleanupSlotBuffers(slot);
    if (!createBuffer(lastInputBufferSize, slot.inputBuffer, slot.inputMemory, &slot.inputPtr, "input")) return false;
    if (!createBuffer(lastOutputBufferSize, slot.outputBuffer, slot.outputMemory, &slot.outputPtr, "output")) return false;

    descriptorSetNeedsUpdate = true;
    return true;
}

bool VulkanUpsampler::createBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr, const char* label) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        printf("[!] Failed to create %s buffer\n", label);
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    
    uint32_t memoryTypeIndex;
    try {
        memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, BUFFER_MEMORY_PROPS);
    } catch (const std::runtime_error& e) {
        printf("[!] %s for %s buffer\n", e.what(), label);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }
    
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    // Store memory properties for flush/invalidate operations
    std::string labelStr(label);
    if (labelStr.find("input") != std::string::npos) {
        inputMemoryProperties = memProps.memoryTypes[memoryTypeIndex].propertyFlags;
    } else if (labelStr.find("output") != std::string::npos) {
        outputMemoryProperties = memProps.memoryTypes[memoryTypeIndex].propertyFlags;
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        printf("[!] Failed to allocate memory for %s buffer\n", label);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        printf("[!] Failed to bind memory for %s buffer\n", label);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    if (vkMapMemory(device, memory, 0, size, 0, mappedPtr) != VK_SUCCESS) {
        printf("[!] Failed to map %s memory\n", label);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    return true;
}

uint32_t VulkanUpsampler::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && 
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

bool VulkanUpsampler::uploadInputToGPU(const float* input, uint32_t totalSamples) {
    if (!slot.inputPtr) [[unlikely]] {
        printf("[!] Input memory not mapped\n");
        return false;
    }

    const VkDeviceSize size = sizeof(float) * totalSamples;
    std::memcpy(slot.inputPtr, input, size);

    // Flush if non-coherent
    if (!(inputMemoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange flushRange{};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = slot.inputMemory;
        flushRange.offset = 0;
        flushRange.size = size;
        
        if (vkFlushMappedMemoryRanges(device, 1, &flushRange) != VK_SUCCESS) {
            printf("[!] Failed to flush input memory\n");
            return false;
        }
    }

    return true;
}

bool VulkanUpsampler::createPipeline(VkShaderModule shader) {
    VkDescriptorSetLayoutBinding bindings[2]{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        printf("[!] Failed to create descriptor set layout\n");
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        printf("[!] Failed to create pipeline layout\n");
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shader;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        printf("[!] Failed to create compute pipeline\n");
        return false;
    }

    return true;
}

bool VulkanUpsampler::createDescriptorSet() {
    if (descriptorSetLayout == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        printf("[!] Descriptor set layout or pipeline layout not initialized\n");
        return false;
    }

    if (descriptorPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pi.poolSizeCount = 1; 
        pi.pPoolSizes = &poolSize; 
        pi.maxSets = 1;
        if (vkCreateDescriptorPool(device, &pi, nullptr, &descriptorPool) != VK_SUCCESS) {
            printf("[!] Failed to create descriptor pool\n"); 
            return false;
        }
    }

    if (descriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descriptorPool; 
        ai.descriptorSetCount = 1; 
        ai.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &ai, &descriptorSet) != VK_SUCCESS) {
            printf("[!] Failed to allocate descriptor set\n"); 
            return false;
        }
        descriptorSetNeedsUpdate = true;
    }

    if (descriptorSetNeedsUpdate) {
        VkDescriptorBufferInfo inInfo{ slot.inputBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outInfo{ slot.outputBuffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outInfo, nullptr };
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        descriptorSetNeedsUpdate = false;
    }

    return true;
}

bool VulkanUpsampler::dispatch(uint32_t inSamples, uint32_t outSamples) {
    if (slot.inFlight) {
        return false;
    }

    VkResult fenceStatus = vkGetFenceStatus(device, slot.fence);
    if (fenceStatus == VK_NOT_READY) {
        return false;
    }

    if (fenceStatus == VK_SUCCESS) {
        vkResetFences(device, 1, &slot.fence);
    }

    if (vkResetCommandBuffer(slot.commandBuffer, 0) != VK_SUCCESS) {
        printf("[!] Failed to reset command buffer\n");
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(slot.commandBuffer, &beginInfo) != VK_SUCCESS) {
        printf("[!] Failed to begin command buffer\n");
        return false;
    }

    vkCmdBindPipeline(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    PushConstants push{};
    push.inFrameCount = inSamples / numChannels;
    push.outFrameCount = outSamples / numChannels;
    push.ratio = static_cast<float>(static_cast<double>(outRate) / inRate);
    vkCmdPushConstants(slot.commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    // Memory barriers for CPU-GPU synchronization
    VkMemoryBarrier inputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    inputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    inputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        slot.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &inputBarrier,
        0, nullptr,
        0, nullptr
    );

    const uint32_t groupCount = (outSamples + VulkanConfig::WORKGROUP_SIZE - 1) / VulkanConfig::WORKGROUP_SIZE;
    vkCmdDispatch(slot.commandBuffer, groupCount, 1, 1);

    VkMemoryBarrier outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        slot.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &outputBarrier,
        0, nullptr,
        0, nullptr
    );

    if (vkEndCommandBuffer(slot.commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to record command buffer\n");
        return false;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submit, slot.fence) != VK_SUCCESS) {
        printf("[!] Failed to submit compute work\n");
        return false;
    }

    slot.inFlight = true;
    return true;
}

bool VulkanUpsampler::enqueue(const float* input, uint32_t inputFrames) {
    if (!slot.initialized || shaderModule == VK_NULL_HANDLE) {
        return false;
    }

    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t inSamples = inputFrames * numChannels;

    // Combine tail from previous frame with current input
    const size_t tailSize = previousTail.size();
    if (tailSize > UINT32_MAX || inSamples > UINT32_MAX - tailSize) {
        return false;
    }

    workingInputBuffer.clear();
    workingInputBuffer.insert(workingInputBuffer.end(), previousTail.begin(), previousTail.end());
    workingInputBuffer.insert(workingInputBuffer.end(), input, input + inSamples);

    const uint32_t fullInSamples = static_cast<uint32_t>(workingInputBuffer.size());
    const uint32_t fullInFrames = fullInSamples / numChannels;
    const uint32_t outFrames = static_cast<uint32_t>(fullInFrames * ratio);
    const uint32_t outSamples = outFrames * numChannels;

    // Calculate skip offset to remove tail's contribution from output
    const uint32_t skipOffset = static_cast<uint32_t>((static_cast<float>(tailSize) / numChannels) * ratio) * numChannels;

    if (!createBuffers(fullInFrames)) {
        return false;
    }

    if (computePipeline == VK_NULL_HANDLE) {
        if (!createPipeline(shaderModule)) {
            return false;
        }
    }
    
    if (descriptorSet == VK_NULL_HANDLE || descriptorSetNeedsUpdate) {
        if (!createDescriptorSet()) {
            return false;
        }
    }

    if (!uploadInputToGPU(workingInputBuffer.data(), fullInSamples)) {
        return false;
    }

    if (!dispatch(fullInSamples, outSamples)) {
        return false;
    }

    slot.sequenceId = nextSequenceId++;
    slot.expectedOutSamples = outSamples;
    slot.skipOffset = skipOffset;

    updateTailBuffer(input, inSamples);

    return true;
}

bool VulkanUpsampler::poll(bool& ready, float* output, uint32_t& outputFrames) {
    ready = false;
    
    if (!slot.inFlight) {
        return true;
    }

    // Use 1ms timeout - proven to work reliably
    // Shorter timeouts (0 or 100us) can cause fence signal issues on some drivers
    const uint64_t timeoutNs = 1000000; // 1ms in nanoseconds
    VkResult waitResult = vkWaitForFences(device, 1, &slot.fence, VK_TRUE, timeoutNs);
    
    if (waitResult == VK_TIMEOUT) {
        // GPU still working, will try again on next poll
        return true;
    }
    
    if (waitResult != VK_SUCCESS) {
        printf("[!] Fence wait error: %d\n", waitResult);
        return false;
    }

    // Download results
    const uint32_t total = slot.expectedOutSamples;
    if (workingOutputBuffer.capacity() < total) {
        workingOutputBuffer.reserve(total * 2);
    }
    workingOutputBuffer.resize(total);

    if (!downloadOutputFromGPU(workingOutputBuffer.data(), total)) {
        return false;
    }

    // Apply skip offset and copy to output
    if (slot.skipOffset > total) {
        printf("[!] Skip offset exceeds total output\n");
        return false;
    }
    
    const uint32_t copySamples = total - slot.skipOffset;
    std::memcpy(output, workingOutputBuffer.data() + slot.skipOffset, sizeof(float) * copySamples);
    outputFrames = copySamples / numChannels;

    slot.inFlight = false;
    ready = true;
    return true;
}

bool VulkanUpsampler::downloadOutputFromGPU(float* output, uint32_t totalSamples) {
    if (!slot.outputPtr) [[unlikely]] {
        printf("[!] Output memory not mapped\n");
        return false;
    }

    // Invalidate cache if non-coherent
    if (!(outputMemoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slot.outputMemory;
        range.offset = 0;
        range.size = sizeof(float) * totalSamples;

        if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
            printf("[!] Failed to invalidate output memory\n");
            return false;
        }
    }

    std::memcpy(output, slot.outputPtr, sizeof(float) * totalSamples);
    return true;
}

void VulkanUpsampler::updateTailBuffer(const float* input, uint32_t inSamples) {
    const uint32_t tailStoreSamples = VulkanConfig::MAX_TAIL_FRAMES * numChannels;
    
    if (inSamples >= tailStoreSamples) {
        previousTail.assign(input + inSamples - tailStoreSamples, input + inSamples);
    } else {
        previousTail.assign(input, input + inSamples);
    }
}

void VulkanUpsampler::cleanupPipeline() {
    if (computePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, computePipeline, nullptr);
        computePipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
        descriptorSet = VK_NULL_HANDLE; // Automatically freed with pool
    }
}

void VulkanUpsampler::cleanupVulkan() {
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

    physicalDevice = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
}

void VulkanUpsampler::cleanupGpuSlot(GpuSlot& cleanupSlot) {
    cleanupSlot.commandBuffer = VK_NULL_HANDLE;

    if (cleanupSlot.fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, cleanupSlot.fence, nullptr);
        cleanupSlot.fence = VK_NULL_HANDLE;
    }

    if (cleanupSlot.inputPtr) {
        vkUnmapMemory(device, cleanupSlot.inputMemory);
        cleanupSlot.inputPtr = nullptr;
    }
    if (cleanupSlot.inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, cleanupSlot.inputBuffer, nullptr);
        cleanupSlot.inputBuffer = VK_NULL_HANDLE;
    }
    if (cleanupSlot.inputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, cleanupSlot.inputMemory, nullptr);
        cleanupSlot.inputMemory = VK_NULL_HANDLE;
    }

    if (cleanupSlot.outputPtr) {
        vkUnmapMemory(device, cleanupSlot.outputMemory);
        cleanupSlot.outputPtr = nullptr;
    }
    if (cleanupSlot.outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, cleanupSlot.outputBuffer, nullptr);
        cleanupSlot.outputBuffer = VK_NULL_HANDLE;
    }
    if (cleanupSlot.outputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, cleanupSlot.outputMemory, nullptr);
        cleanupSlot.outputMemory = VK_NULL_HANDLE;
    }

    cleanupSlot.initialized = false;
}

void VulkanUpsampler::cleanupSlotBuffers(GpuSlot& slotRef) {
    if (slotRef.inputPtr) {
        vkUnmapMemory(device, slotRef.inputMemory);
        slotRef.inputPtr = nullptr;
    }
    if (slotRef.inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, slotRef.inputBuffer, nullptr);
        slotRef.inputBuffer = VK_NULL_HANDLE;
    }
    if (slotRef.inputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, slotRef.inputMemory, nullptr);
        slotRef.inputMemory = VK_NULL_HANDLE;
    }

    if (slotRef.outputPtr) {
        vkUnmapMemory(device, slotRef.outputMemory);
        slotRef.outputPtr = nullptr;
    }
    if (slotRef.outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, slotRef.outputBuffer, nullptr);
        slotRef.outputBuffer = VK_NULL_HANDLE;
    }
    if (slotRef.outputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, slotRef.outputMemory, nullptr);
        slotRef.outputMemory = VK_NULL_HANDLE;
    }
}
