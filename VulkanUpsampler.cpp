#include "VulkanUpsampler.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// === Configuration Constants ===
namespace VulkanConfig {
    static constexpr uint32_t WORKGROUP_SIZE = 64;
    static constexpr uint32_t MAX_TAIL_FRAMES = 4;
    static constexpr uint32_t API_VERSION = VK_API_VERSION_1_2;
    static constexpr float QUEUE_PRIORITY = 1.0f;
    static constexpr const char* APP_NAME = "VulkanUpsampler";
    static constexpr const char* ENGINE_NAME = "No Engine";
    static constexpr uint64_t FENCE_TIMEOUT = UINT64_MAX; // Infinite timeout
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

    // === Allocate single GpuSlot ===
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(device, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS) {
        printf("[!] Failed to create slot fence\n");
        return false;
    }

    // Allocate command buffer for slot
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cmdAlloc, &slot.commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to allocate slot command buffer\n");
        return false;
    }

    // Allocate input buffer
    VkDeviceSize inputSize = 4096 * sizeof(float); // conservative
    if (!createBuffer(inputSize, slot.inputBuffer, slot.inputMemory, &slot.inputPtr, "slot.input")) {
        return false;
    }

    // Allocate output buffer
    VkDeviceSize outputSize = 8192 * sizeof(float); // conservative
    if (!createBuffer(outputSize, slot.outputBuffer, slot.outputMemory, &slot.outputPtr, "slot.output")) {
        return false;
    }

    slot.initialized = true;

    // === Pre-allocate working buffers based on expected usage ===
    const float ratio = static_cast<float>(outputRate) / inputRate;
    const uint32_t estimatedInputSamples = 2048 * channels;  // Typical frame size
    const uint32_t estimatedOutputSamples = static_cast<uint32_t>(estimatedInputSamples * ratio);
    
    workingInputBuffer.reserve(estimatedInputSamples * 2);   // Extra headroom
    workingOutputBuffer.reserve(estimatedOutputSamples * 2); // Extra headroom
    
    printf("[+] VulkanUpsampler initialized: %uHz -> %uHz (%u channels)\n", 
           inputRate, outputRate, channels);
    printf("[*] Pre-allocated buffers: input=%zu, output=%zu samples\n",
           workingInputBuffer.capacity(), workingOutputBuffer.capacity());

    return true;
}

void VulkanUpsampler::setKernel(ResampleKernel kernel) {
    selectedKernel = kernel;

    // Clean up previous shader module
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }

    // Map kernel enum to shader filename
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
    if (shaderModule == VK_NULL_HANDLE) [[unlikely]] {
        printf("[!] Shader module not loaded\n");
        return false;
    }

    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t inSamples = inputFrames * numChannels;

    // === Safe size conversion ===
    const size_t tailSize = previousTail.size();
    
    // Size check (prevent 32-bit overflow)
    if (tailSize > UINT32_MAX || inSamples > UINT32_MAX - tailSize) [[unlikely]] {
        printf("[!] Buffer size exceeds 32-bit limit: tail=%zu, input=%u\n", tailSize, inSamples);
        return false;
    }
    
    const uint32_t totalInputSamples = static_cast<uint32_t>(tailSize) + inSamples;
    
    // Reserve space if needed (avoid frequent reallocations)
    if (workingInputBuffer.capacity() < totalInputSamples) {
        workingInputBuffer.reserve(totalInputSamples * 2);  // Reserve extra for future growth
        printf("[*] Input buffer capacity increased to %zu samples\n", workingInputBuffer.capacity());
    }
    
    // Efficiently prepare input data
    workingInputBuffer.clear();
    workingInputBuffer.insert(workingInputBuffer.end(), previousTail.begin(), previousTail.end());
    workingInputBuffer.insert(workingInputBuffer.end(), input, input + inSamples);

    // === Safe size conversion (workingInputBuffer.size()) ===
    const size_t bufferSize = workingInputBuffer.size();
    if (bufferSize > UINT32_MAX) [[unlikely]] {
        printf("[!] Working buffer size exceeds 32-bit limit: %zu\n", bufferSize);
        return false;
    }
    
    const uint32_t fullInSamples = static_cast<uint32_t>(bufferSize);
    const uint32_t fullInFrames = fullInSamples / numChannels;

    // Calculate buffer requirements
    const uint32_t maxExpectedInFrames = inputFrames + VulkanConfig::MAX_TAIL_FRAMES;

    // GPU processing pipeline
    if (!createBuffers(maxExpectedInFrames)) return false;
    if (!uploadInputToGPU(workingInputBuffer.data(), fullInSamples)) return false;
    
    if (computePipeline == VK_NULL_HANDLE) {
        if (!createPipeline(shaderModule)) return false;
    }
    
    if (!createDescriptorSet()) return false;

    // Execute GPU computation
    const uint32_t actualOutFrames = static_cast<uint32_t>(fullInFrames * ratio);
    const uint32_t actualOutSamples = actualOutFrames * numChannels;
    
    if (!dispatch(fullInSamples, actualOutSamples)) return false;

    // === Optimized output buffer management ===
    // Reserve space for output if needed
    if (workingOutputBuffer.capacity() < actualOutSamples) {
        workingOutputBuffer.reserve(actualOutSamples * 2);  // Reserve extra for future growth
        printf("[*] Output buffer capacity increased to %zu samples\n", workingOutputBuffer.capacity());
    }
    
    workingOutputBuffer.resize(actualOutSamples);
    if (!downloadOutputFromGPU(workingOutputBuffer.data(), actualOutSamples)) return false;

    // Calculate output offset based on tail processing
    const float tailFramesF = static_cast<float>(tailSize) / numChannels;  // Modified: previousTail.size() -> tailSize
    const float skipFramesF = tailFramesF * ratio;
    const uint32_t skipOffset = static_cast<uint32_t>(skipFramesF * numChannels);

    // === Safe size conversion (workingOutputBuffer.size()) ===
    const size_t outputSize = workingOutputBuffer.size();
    if (skipOffset > outputSize) [[unlikely]] {
        printf("[!] Output calculation error: skip=%u > total=%zu\n", skipOffset, outputSize);
        return false;
    }

    // Copy final output - Safe conversion
    if (outputSize > UINT32_MAX) [[unlikely]] {
        printf("[!] Output buffer size exceeds 32-bit limit: %zu\n", outputSize);
        return false;
    }
    
    const uint32_t outCopyCount = static_cast<uint32_t>(outputSize) - skipOffset;
    std::memcpy(output, workingOutputBuffer.data() + skipOffset, outCopyCount * sizeof(float));
    outputFrames = outCopyCount / numChannels;

    // Update tail buffer for next iteration
    updateTailBuffer(input, inSamples);

    return true;
}

void VulkanUpsampler::shutdown() {
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

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    
    // Flag change for pool reset optimization
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;  // For short-lived command buffers

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

    printf("[+] Shader module created: %s (%.1f KB)\n", filename.c_str(), fileSize / 1024.0);
    return module;
}

bool VulkanUpsampler::createBuffers(uint32_t inputFrames) {
    const uint32_t totalInputSamples = inputFrames * numChannels;
    const uint32_t totalOutputSamples = static_cast<uint32_t>(inputFrames * (static_cast<float>(outRate) / inRate)) * numChannels;

    const VkDeviceSize requiredInputSize = sizeof(float) * totalInputSamples;
    const VkDeviceSize requiredOutputSize = sizeof(float) * totalOutputSamples;

    // Check if resize is necessary
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
        return true; // Existing buffers are sufficient
    }

    // Only destroy slot buffer resources (not full slot)
    cleanupSlotBuffers(slot);

    if (!createBuffer(lastInputBufferSize, slot.inputBuffer, slot.inputMemory, &slot.inputPtr, "input")) return false;
    if (!createBuffer(lastOutputBufferSize, slot.outputBuffer, slot.outputMemory, &slot.outputPtr, "output")) return false;

    descriptorSetNeedsUpdate = true;

    printf("[+] GPU buffers created: input=%.1f KB, output=%.1f KB (frames=%u)\n",
        requiredInputSize / 1024.0, requiredOutputSize / 1024.0, inputFrames);

    return true;
}

bool VulkanUpsampler::createBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr, const char* label) {
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        printf("[!] Failed to create %s buffer\n", label);
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    // Get memory properties (needed for storing memory properties)
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    // Allocate memory
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
    
    // Store memory properties for later use (IMPORTANT: this was missing!)
    if (std::string(label) == "input") {
        inputMemoryProperties = memProps.memoryTypes[memoryTypeIndex].propertyFlags;
    } else if (std::string(label) == "output") {
        outputMemoryProperties = memProps.memoryTypes[memoryTypeIndex].propertyFlags;
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        printf("[!] Failed to allocate memory for %s buffer (size: %llu bytes)\n",
            label, static_cast<unsigned long long>(allocInfo.allocationSize));
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    // Bind memory to buffer
    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        printf("[!] Failed to bind memory for %s buffer\n", label);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    // Map memory for persistent access
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

    // Flush if memory is not coherent
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
    // Create descriptor set layout
    VkDescriptorSetLayoutBinding bindings[2]{};

    // Input buffer binding
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Output buffer binding  
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

    // Create push constant range
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    // Create pipeline layout
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

    // Create shader stage
    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shader;
    shaderStage.pName = "main";

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        printf("[!] Failed to create compute pipeline\n");
        return false;
    }

    printf("[+] Compute pipeline created successfully\n");
    return true;
}

bool VulkanUpsampler::createDescriptorSet() {
    if (descriptorSetLayout == VK_NULL_HANDLE) [[unlikely]] {
        printf("[!] Descriptor set layout not created\n");
        return false;
    }

    // Create descriptor pool if needed
    if (descriptorPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            printf("[!] Failed to create descriptor pool\n");
            return false;
        }
    }

    // Allocate descriptor set if needed
    if (descriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            printf("[!] Failed to allocate descriptor set\n");
            return false;
        }
        descriptorSetNeedsUpdate = true;  // Update needed when newly created
    }

    // Optimization: Update only when buffers have changed
    if (descriptorSetNeedsUpdate) {
        VkDescriptorBufferInfo inputInfo{};
        inputInfo.buffer = slot.inputBuffer;
        inputInfo.offset = 0;
        inputInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo outputInfo{};
        outputInfo.buffer = slot.outputBuffer;
        outputInfo.offset = 0;
        outputInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inputInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outputInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        descriptorSetNeedsUpdate = false;  // Update completed
        printf("[*] Descriptor set updated\n");
    }

    return true;
}

bool VulkanUpsampler::dispatch(uint32_t inSamples, uint32_t outSamples) {
    if (!slot.initialized) {
        printf("[!] GpuSlot not initialized\n");
        return false;
    }

    if (computePipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) [[unlikely]] {
        printf("[!] Pipeline not ready for dispatch\n");
        return false;
    }

    // === Wait for previous work to complete ===
    VkResult fenceStatus = vkGetFenceStatus(device, slot.fence);
    if (fenceStatus == VK_NOT_READY) {
        if (vkWaitForFences(device, 1, &slot.fence, VK_TRUE, VulkanConfig::FENCE_TIMEOUT) != VK_SUCCESS) {
            printf("[!] Failed to wait for previous GPU work\n");
            return false;
        }
    }

    vkResetFences(device, 1, &slot.fence);

    // === Reset command buffer ===
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

    // === Record compute commands ===
    vkCmdBindPipeline(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(slot.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    PushConstants push{};
    push.inFrameCount = inSamples / numChannels;
    push.outFrameCount = outSamples / numChannels;
    push.ratio = static_cast<float>(static_cast<double>(outRate) / inRate);

    vkCmdPushConstants(slot.commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);

    // Memory barrier for input buffer
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

    const uint32_t totalSampleThreads = outSamples;
    const uint32_t groupCount = (totalSampleThreads + VulkanConfig::WORKGROUP_SIZE - 1) / VulkanConfig::WORKGROUP_SIZE;

    vkCmdDispatch(slot.commandBuffer, groupCount, 1, 1);

    // Memory barrier for output buffer
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

    // === Submit ===
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submit, slot.fence) != VK_SUCCESS) {
        printf("[!] Failed to submit compute work\n");
        return false;
    }

    // === Wait (synchronous) ===
    const VkResult waitResult = vkWaitForFences(device, 1, &slot.fence, VK_TRUE, VulkanConfig::FENCE_TIMEOUT);
    if (waitResult != VK_SUCCESS) {
        printf("[!] GPU dispatch timeout or error\n");
        return false;
    }

    return true;
}

bool VulkanUpsampler::downloadOutputFromGPU(float* output, uint32_t totalSamples) {
    if (!slot.outputPtr) [[unlikely]] {
        printf("[!] Output memory not mapped\n");
        return false;
    }

    // Invalidate cache if memory is not coherent
    if (!(outputMemoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slot.outputMemory;
        range.offset = 0;
        range.size = sizeof(float) * totalSamples;

        if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
            printf("[!] Failed to invalidate output memory range\n");
            return false;
        }
    }

    std::memcpy(output, slot.outputPtr, sizeof(float) * totalSamples);
    return true;
}

void VulkanUpsampler::updateTailBuffer(const float* input, uint32_t inSamples) {
    const uint32_t tailStoreSamples = VulkanConfig::MAX_TAIL_FRAMES * numChannels;
    
    if (inSamples >= tailStoreSamples) {
        // Store last portion of input as new tail
        previousTail.assign(input + inSamples - tailStoreSamples, input + inSamples);
    } else {
        // Store entire input as tail (input is smaller than tail buffer)
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

    // Reset all handles
    physicalDevice = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
}

void VulkanUpsampler::cleanupGpuSlot(GpuSlot& cleanupSlot) {
    if (cleanupSlot.commandBuffer != VK_NULL_HANDLE) {
        // Command buffer는 pool에서 일괄 해제됨 (생략 가능)
        cleanupSlot.commandBuffer = VK_NULL_HANDLE;
    }

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
