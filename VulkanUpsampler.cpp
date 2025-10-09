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

    // Initialize all slots
    VkDeviceSize inputSize = 4096 * sizeof(float);
    VkDeviceSize outputSize = 8192 * sizeof(float);

    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        // Create fence for async synchronization
        VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(device, &fenceInfo, nullptr, &slots[i].fence) != VK_SUCCESS) {
            printf("[!] Failed to create fence for slot %u\n", i);
            return false;
        }

        // Allocate initial GPU buffers
        if (!createBuffer(inputSize, slots[i].inputBuffer, slots[i].inputMemory, &slots[i].inputPtr, "slot.input")) {
            printf("[!] Failed to create input buffer for slot %u\n", i);
            return false;
        }

        if (!createBuffer(outputSize, slots[i].outputBuffer, slots[i].outputMemory, &slots[i].outputPtr, "slot.output")) {
            printf("[!] Failed to create output buffer for slot %u\n", i);
            return false;
        }

        slots[i].inputBufferSize = inputSize;
        slots[i].outputBufferSize = outputSize;
        slots[i].descriptorNeedsUpdate = true;
        slots[i].initialized = true;
    }

    // Pre-allocate working buffers
    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t estimatedInputSamples = 2048 * numChannels;
    const uint32_t estimatedOutputSamples = static_cast<uint32_t>(estimatedInputSamples * ratio);
    
    workingInputBuffer.reserve(estimatedInputSamples * 2);
    workingOutputBuffer.reserve(estimatedOutputSamples * 2);
    
    printf("[+] VulkanUpsampler initialized: %uHz -> %uHz (%u channels, %u slots)\n", 
           inputRate, outputRate, channels, NUM_SLOTS);
    
    return true;
}

void VulkanUpsampler::setKernel(ResampleKernel kernel) {
    // If kernel hasn't changed and pipeline already exists, skip recreation
    if (kernel == selectedKernel && computePipeline != VK_NULL_HANDLE) {
        return;
    }

    selectedKernel = kernel;

    // Clean up old shader and pipeline if they exist
    if (computePipeline != VK_NULL_HANDLE) {
        cleanupPipeline();
    }
    
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }

    // Load new shader
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
        return;
    }

    // Create pipeline with new shader
    if (!createPipeline(shaderModule)) {
        printf("[!] Failed to create compute pipeline\n");
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
        return;
    }

    // Recreate descriptor sets for all slots with new pipeline
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        // Mark descriptor as needing update
        slots[i].descriptorNeedsUpdate = true;
        
        if (!createDescriptorSet(i)) {
            printf("[!] Failed to create descriptor set for slot %u\n", i);
            cleanupPipeline();
            vkDestroyShaderModule(device, shaderModule, nullptr);
            shaderModule = VK_NULL_HANDLE;
            return;
        }
    }
    
    printf("[+] Pipeline ready (%s kernel)\n", filename.c_str());
}

bool VulkanUpsampler::process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) {
    if (!enqueue(input, inputFrames)) {
        return false;
    }
    
    bool ready = false;
    // First, try non-blocking polls
    for (int i = 0; i < 1000 && !ready; ++i) {
        if (!poll(ready, output, outputFrames)) {
            return false;
        }
        
        // Early exit if ready
        if (ready) {
            return true;
        }
        
        // Small yield to prevent tight spin loop
        if (i % 10 == 0) {
            std::this_thread::yield();
        }
    }
    
    // If still not ready after 1000 iterations, do a blocking wait
    if (!ready && !pendingQueue.empty()) {
        const PendingWork& work = pendingQueue.front();
        GpuSlot& slot = slots[work.slotIndex];
        
        if (slot.inFlight && slot.fence != VK_NULL_HANDLE) {
            // Final blocking wait with timeout
            const uint64_t timeoutNs = 10000000; // 10ms timeout
            VkResult waitResult = vkWaitForFences(device, 1, &slot.fence, VK_TRUE, timeoutNs);
            
            if (waitResult == VK_SUCCESS) {
                // Try poll one more time
                if (!poll(ready, output, outputFrames)) {
                    return false;
                }
            }
        }
    }
    
    if (!ready) {
        printf("[!] GPU processing timeout\n");
        return false;
    }
    
    return true;
}

void VulkanUpsampler::shutdown() {
    // Wait for all in-flight GPU work to complete before cleanup
    if (device != VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
            if (slots[i].inFlight && slots[i].fence != VK_NULL_HANDLE) {
                vkWaitForFences(device, 1, &slots[i].fence, VK_TRUE, VulkanConfig::FENCE_TIMEOUT);
                slots[i].inFlight = false;
            }
        }

        // Clean up all slots
        for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
            if (slots[i].initialized) {
                cleanupGpuSlot(slots[i]);
            }
        }
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
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        printf("[!] Failed to create command pool\n");
        return false;
    }

    // Allocate command buffers for all slots
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = NUM_SLOTS;

    std::array<VkCommandBuffer, NUM_SLOTS> commandBuffers;
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        printf("[!] Failed to allocate command buffers\n");
        return false;
    }

    // Assign command buffers to slots
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        slots[i].commandBuffer = commandBuffers[i];
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

bool VulkanUpsampler::createBuffers(uint32_t inputFrames, uint32_t slotIndex) {
    if (slotIndex >= NUM_SLOTS) {
        printf("[!] Invalid slot index: %u\n", slotIndex);
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t totalInputSamples = inputFrames * numChannels;
    const uint32_t totalOutputSamples = static_cast<uint32_t>(inputFrames * ratio) * numChannels;

    const VkDeviceSize requiredInputSize = sizeof(float) * totalInputSamples;
    const VkDeviceSize requiredOutputSize = sizeof(float) * totalOutputSamples;

    // Check if resize needed for this slot
    bool needsResize = false;
    if (requiredInputSize > slot.inputBufferSize) {
        slot.inputBufferSize = requiredInputSize;
        needsResize = true;
    }
    if (requiredOutputSize > slot.outputBufferSize) {
        slot.outputBufferSize = requiredOutputSize;
        needsResize = true;
    }

    if (!needsResize) {
        return true;
    }

    // Recreate buffers for this slot
    cleanupSlotBuffers(slot);
    if (!createBuffer(slot.inputBufferSize, slot.inputBuffer, slot.inputMemory, &slot.inputPtr, "input")) return false;
    if (!createBuffer(slot.outputBufferSize, slot.outputBuffer, slot.outputMemory, &slot.outputPtr, "output")) return false;

    slot.descriptorNeedsUpdate = true;
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

bool VulkanUpsampler::uploadInputToGPU(const float* input, uint32_t totalSamples, uint32_t slotIndex) {
    if (slotIndex >= NUM_SLOTS) {
        printf("[!] Invalid slot index: %u\n", slotIndex);
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    if (!slot.inputPtr) [[unlikely]] {
        printf("[!] Input memory not mapped for slot %u\n", slotIndex);
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
            printf("[!] Failed to flush input memory for slot %u\n", slotIndex);
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

bool VulkanUpsampler::createDescriptorSet(uint32_t slotIndex) {
    if (descriptorSetLayout == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        printf("[!] Descriptor set layout or pipeline layout not initialized\n");
        return false;
    }

    if (slotIndex >= NUM_SLOTS) {
        printf("[!] Invalid slot index: %u\n", slotIndex);
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    // Create descriptor pool if not exists (shared across all slots)
    if (descriptorPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * NUM_SLOTS };
        VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pi.poolSizeCount = 1; 
        pi.pPoolSizes = &poolSize; 
        pi.maxSets = NUM_SLOTS;
        if (vkCreateDescriptorPool(device, &pi, nullptr, &descriptorPool) != VK_SUCCESS) {
            printf("[!] Failed to create descriptor pool\n"); 
            return false;
        }
    }

    // Allocate descriptor set for this slot if not exists
    if (slot.descriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descriptorPool; 
        ai.descriptorSetCount = 1; 
        ai.pSetLayouts = &descriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &ai, &slot.descriptorSet) != VK_SUCCESS) {
            printf("[!] Failed to allocate descriptor set for slot %u\n", slotIndex); 
            return false;
        }
        slot.descriptorNeedsUpdate = true;
    }

    // Update descriptor set if needed
    if (slot.descriptorNeedsUpdate) {
        VkDescriptorBufferInfo inInfo{ slot.inputBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outInfo{ slot.outputBuffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, slot.descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, slot.descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outInfo, nullptr };
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        slot.descriptorNeedsUpdate = false;
    }

    return true;
}

bool VulkanUpsampler::dispatch(uint32_t inSamples, uint32_t outSamples, uint32_t slotIndex) {
    if (slotIndex >= NUM_SLOTS) {
        printf("[!] Invalid slot index: %u\n", slotIndex);
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    // Ensure slot is not in-flight
    if (slot.inFlight) {
        printf("[!] Slot %u still in-flight\n", slotIndex);
        return false;
    }

    // Wait for fence if it was previously used (should be signaled already)
    if (slot.fence != VK_NULL_HANDLE) {
        VkResult fenceStatus = vkGetFenceStatus(device, slot.fence);
        if (fenceStatus == VK_NOT_READY) {
            printf("[!] Slot %u fence not ready (should not happen)\n", slotIndex);
            return false;
        }
        
        // Reset fence for reuse
        if (vkResetFences(device, 1, &slot.fence) != VK_SUCCESS) {
            printf("[!] Failed to reset fence for slot %u\n", slotIndex);
            return false;
        }
    }

    // Reset command buffer
    if (vkResetCommandBuffer(slot.commandBuffer, 0) != VK_SUCCESS) {
        printf("[!] Failed to reset command buffer for slot %u\n", slotIndex);
        return false;
    }

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(slot.commandBuffer, &beginInfo) != VK_SUCCESS) {
        printf("[!] Failed to begin command buffer for slot %u\n", slotIndex);
        return false;
    }

    // Bind pipeline and descriptor sets
    vkCmdBindPipeline(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &slot.descriptorSet, 0, nullptr);

    // Set push constants
    PushConstants push{};
    push.inFrameCount = inSamples / numChannels;
    push.outFrameCount = outSamples / numChannels;
    push.ratio = static_cast<float>(static_cast<double>(outRate) / inRate);
    vkCmdPushConstants(slot.commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    // Buffer-specific memory barriers for better performance (vs global VkMemoryBarrier)
    VkBufferMemoryBarrier inputBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    inputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    inputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    inputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inputBarrier.buffer = slot.inputBuffer;
    inputBarrier.offset = 0;
    inputBarrier.size = sizeof(float) * inSamples;

    vkCmdPipelineBarrier(
        slot.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &inputBarrier,
        0, nullptr
    );

    // Dispatch compute work
    const uint32_t groupCount = (outSamples + VulkanConfig::WORKGROUP_SIZE - 1) / VulkanConfig::WORKGROUP_SIZE;
    vkCmdDispatch(slot.commandBuffer, groupCount, 1, 1);

    // Output barrier
    VkBufferMemoryBarrier outputBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.buffer = slot.outputBuffer;
    outputBarrier.offset = 0;
    outputBarrier.size = sizeof(float) * outSamples;

    vkCmdPipelineBarrier(
        slot.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, nullptr,
        1, &outputBarrier,
        0, nullptr
    );

    // End command buffer
    if (vkEndCommandBuffer(slot.commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to record command buffer for slot %u\n", slotIndex);
        return false;
    }

    // Submit to queue with fence
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submit, slot.fence) != VK_SUCCESS) {
        printf("[!] Failed to submit compute work for slot %u\n", slotIndex);
        return false;
    }

    // Mark slot as in-flight
    slot.inFlight = true;
    return true;
}

bool VulkanUpsampler::enqueue(const float* input, uint32_t inputFrames) {
    if (shaderModule == VK_NULL_HANDLE) {
        return false;
    }

    if (computePipeline == VK_NULL_HANDLE) {
        printf("[!] Pipeline not initialized\n");
        return false;
    }

    // Find available slot
    int slotIndex = findAvailableSlot();
    if (slotIndex < 0) {
        // All slots busy
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    if (!slot.initialized) {
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

    if (!createBuffers(fullInFrames, slotIndex)) {
        return false;
    }
    
    if (slot.descriptorSet == VK_NULL_HANDLE || slot.descriptorNeedsUpdate) {
        if (!createDescriptorSet(slotIndex)) {
            return false;
        }
    }

    if (!uploadInputToGPU(workingInputBuffer.data(), fullInSamples, slotIndex)) {
        return false;
    }

    if (!dispatch(fullInSamples, outSamples, slotIndex)) {
        return false;
    }

    // Track submission order
    slot.sequenceId = nextSequenceId++;
    slot.expectedOutSamples = outSamples;
    slot.skipOffset = skipOffset;

    pendingQueue.push({ static_cast<uint32_t>(slotIndex), slot.sequenceId });

    // Update round-robin index
    currentSlotIndex = (slotIndex + 1) % NUM_SLOTS;

    updateTailBuffer(input, inSamples);

    return true;
}

bool VulkanUpsampler::poll(bool& ready, float* output, uint32_t& outputFrames) {
    ready = false;
    
    // Check if there are any pending works
    if (pendingQueue.empty()) {
        return true;
    }

    // Get the oldest pending work (FIFO order)
    const PendingWork& work = pendingQueue.front();
    const uint32_t slotIndex = work.slotIndex;
    GpuSlot& slot = slots[slotIndex];
    
    if (!slot.inFlight) {
        // Slot already processed - pop and continue
        pendingQueue.pop();
        return true;
    }

    // Non-blocking fence status check - immediately returns current state
    VkResult fenceStatus = vkGetFenceStatus(device, slot.fence);
    
    if (fenceStatus == VK_NOT_READY) {
        // GPU still working - return immediately without blocking
        return true;
    }
    
    if (fenceStatus != VK_SUCCESS) {
        printf("[!] Fence status error for slot %u: %d\n", slotIndex, fenceStatus);
        slot.inFlight = false;
        pendingQueue.pop();
        return false;
    }

    // Fence signaled - GPU work completed, download results
    const uint32_t total = slot.expectedOutSamples;
    if (workingOutputBuffer.capacity() < total) {
        workingOutputBuffer.reserve(total * 2);
    }
    workingOutputBuffer.resize(total);

    if (!downloadOutputFromGPU(workingOutputBuffer.data(), total, slotIndex)) {
        slot.inFlight = false;
        pendingQueue.pop();
        return false;
    }

    // Apply skip offset and copy to output
    if (slot.skipOffset > total) {
        printf("[!] Skip offset %u exceeds total output %u for slot %u\n", slot.skipOffset, total, slotIndex);
        slot.inFlight = false;
        pendingQueue.pop();
        return false;
    }
    
    const uint32_t copySamples = total - slot.skipOffset;
    std::memcpy(output, workingOutputBuffer.data() + slot.skipOffset, sizeof(float) * copySamples);
    outputFrames = copySamples / numChannels;

    // Clear slot state - work complete
    slot.inFlight = false;
    slot.sequenceId = 0;
    slot.expectedOutSamples = 0;
    slot.skipOffset = 0;
    
    pendingQueue.pop();
    ready = true;
    return true;
}

// === Asynchronous API Implementation ===

uint64_t VulkanUpsampler::processAsync(const float* input, uint32_t inputFrames, CompletionCallback callback) {
    // Check if shader is loaded - silently fail during early initialization
    if (shaderModule == VK_NULL_HANDLE) [[unlikely]] {
        return 0;
    }
    
    if (!enqueue(input, inputFrames)) {
        // Silently fail - this is normal during initialization or when slots are busy
        return 0;
    }
    
    // Get the most recently submitted work
    if (!pendingQueue.empty()) {
        const PendingWork& work = pendingQueue.back();
        GpuSlot& slot = slots[work.slotIndex];
        
        // Store callback for later invocation
        if (callback) {
            slot.callback = callback;
        }
        
        return work.sequenceId;
    }
    
    return 0;
}

size_t VulkanUpsampler::tryPollAll(std::vector<AsyncResult>& results) {
    size_t completedCount = 0;
    
    // Check all pending works (FIFO order)
    while (!pendingQueue.empty()) {
        const PendingWork& work = pendingQueue.front();
        const uint32_t slotIndex = work.slotIndex;
        GpuSlot& slot = slots[slotIndex];
        
        if (!slot.inFlight) {
            // Already processed
            pendingQueue.pop();
            continue;
        }
        
        // Non-blocking check
        VkResult fenceStatus = vkGetFenceStatus(device, slot.fence);
        
        if (fenceStatus == VK_NOT_READY) {
            // Still processing - stop checking (maintain FIFO order)
            break;
        }
        
        if (fenceStatus != VK_SUCCESS) {
            printf("[!] Fence status error for slot %u: %d\n", slotIndex, fenceStatus);
            
            AsyncResult errorResult;
            errorResult.sequenceId = work.sequenceId;
            errorResult.success = false;
            errorResult.frameCount = 0;
            results.push_back(errorResult);
            
            slot.inFlight = false;
            slot.sequenceId = 0;
            slot.expectedOutSamples = 0;
            slot.skipOffset = 0;
            
            pendingQueue.pop();
            continue;
        }
        
        // Work completed - download results
        const uint32_t total = slot.expectedOutSamples;
        
        if (workingOutputBuffer.capacity() < total) {
            workingOutputBuffer.reserve(total * 2);
        }
        workingOutputBuffer.resize(total);
        
        if (!downloadOutputFromGPU(workingOutputBuffer.data(), total, slotIndex)) {
            AsyncResult errorResult;
            errorResult.sequenceId = work.sequenceId;
            errorResult.success = false;
            errorResult.frameCount = 0;
            results.push_back(errorResult);
            
            slot.inFlight = false;
            slot.sequenceId = 0;
            slot.expectedOutSamples = 0;
            slot.skipOffset = 0;
            
            pendingQueue.pop();
            continue;
        }
        
        // Apply skip offset
        if (slot.skipOffset > total) {
            printf("[!] Skip offset %u exceeds total output %u for slot %u\n", slot.skipOffset, total, slotIndex);
            
            AsyncResult errorResult;
            errorResult.sequenceId = work.sequenceId;
            errorResult.success = false;
            errorResult.frameCount = 0;
            results.push_back(errorResult);
            
            slot.inFlight = false;
            slot.sequenceId = 0;
            slot.expectedOutSamples = 0;
            slot.skipOffset = 0;
            
            pendingQueue.pop();
            continue;
        }
        
        const uint32_t copySamples = total - slot.skipOffset;
        const uint32_t outputFrames = copySamples / numChannels;
        
        // Create result
        AsyncResult result;
        result.sequenceId = work.sequenceId;
        result.data.resize(copySamples);
        std::memcpy(result.data.data(), workingOutputBuffer.data() + slot.skipOffset, sizeof(float) * copySamples);
        result.frameCount = outputFrames;
        result.success = true;
        
        results.push_back(result);
        
        // Invoke callback if registered
        if (slot.callback) {
            slot.callback(result.data.data(), outputFrames);
            slot.callback = nullptr; // Clear after invocation
        }
        
        // Clear slot state - work complete
        slot.inFlight = false;
        slot.sequenceId = 0;
        slot.expectedOutSamples = 0;
        slot.skipOffset = 0;
        
        pendingQueue.pop();
        ++completedCount;
    }
    
    return completedCount;
}

size_t VulkanUpsampler::getAvailableSlots() const {
    size_t available = 0;
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        if (!slots[i].inFlight) {
            ++available;
        }
    }
    return available;
}

bool VulkanUpsampler::downloadOutputFromGPU(float* output, uint32_t totalSamples, uint32_t slotIndex) {
    if (slotIndex >= NUM_SLOTS) {
        printf("[!] Invalid slot index: %u\n", slotIndex);
        return false;
    }

    GpuSlot& slot = slots[slotIndex];

    if (!slot.outputPtr) [[unlikely]] {
        printf("[!] Output memory not mapped for slot %u\n", slotIndex);
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
            printf("[!] Failed to invalidate output memory for slot %u\n", slotIndex);
            return false;
        }
    }

    std::memcpy(output, slot.outputPtr, sizeof(float) * totalSamples);
    return true;
}

// === Utility Functions ===
void VulkanUpsampler::updateTailBuffer(const float* input, uint32_t inSamples) {
    const uint32_t tailStoreSamples = VulkanConfig::MAX_TAIL_FRAMES * numChannels;
    
    if (inSamples >= tailStoreSamples) {
        previousTail.assign(input + inSamples - tailStoreSamples, input + inSamples);
    } else {
        previousTail.assign(input, input + inSamples);
    }
}

int VulkanUpsampler::findAvailableSlot() const {
    // First, try round-robin from currentSlotIndex to find a slot not in flight
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        const uint32_t slotIndex = (currentSlotIndex + i) % NUM_SLOTS;
        if (!slots[slotIndex].inFlight) {
            return static_cast<int>(slotIndex);
        }
    }

    // All slots marked in-flight - check for completed work using non-blocking fence query
    for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
        const uint32_t slotIndex = (currentSlotIndex + i) % NUM_SLOTS;
        const GpuSlot& slot = slots[slotIndex];
        
        if (slot.fence != VK_NULL_HANDLE && slot.inFlight) {
            VkResult status = vkGetFenceStatus(device, slot.fence);
            if (status == VK_SUCCESS) {
                // Note: We don't clear inFlight here - that's done by poll() to maintain ordering
                // But we can inform the caller this slot will be available soon
                // For now, we skip this slot and let poll() handle it properly
                continue;
            }
        }
    }

    return -1; // All slots genuinely busy
}

// ...existing cleanup functions...

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
        
        // Descriptor sets are automatically freed with pool
        for (uint32_t i = 0; i < NUM_SLOTS; ++i) {
            slots[i].descriptorSet = VK_NULL_HANDLE;
        }
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
