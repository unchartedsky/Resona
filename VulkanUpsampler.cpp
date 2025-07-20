#include "VulkanUpsampler.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

bool VulkanUpsampler::initialize(uint32_t inputRate, uint32_t outputRate, uint32_t channels) {
    inRate = inputRate;
    outRate = outputRate;
    numChannels = channels;

    if (!initVulkan()) {
        printf("[!] Vulkan initialization failed\n");
        return false;
    }

    return true;
}

void VulkanUpsampler::setKernel(ResampleKernel kernel) {
    selectedKernel = kernel;

    // 이전 shaderModule 정리
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
        printf("[!] Unknown kernel selected\n");
        return;
    }

    shaderModule = createShaderModule(filename);
}

bool VulkanUpsampler::process(const float* input, uint32_t inputFrames, float* output, uint32_t& outputFrames) {
    if (shaderModule == VK_NULL_HANDLE) {
        printf("[!] Shader module not loaded\n");
        return false;
    }

    const float ratio = static_cast<float>(outRate) / inRate;
    const uint32_t inSamples = inputFrames * numChannels;
    const uint32_t outFrames = static_cast<uint32_t>(inputFrames * ratio);
    const uint32_t outSamples = outFrames * numChannels;

	// Step 1: prepare GPU buffers
    if (!createBuffers(inputFrames)) return false;

	// Step 2: copy input data to GPU
    if (!uploadInputToGPU(input, inSamples)) return false;

	// Step 3: create compute pipeline
    if (computePipeline == VK_NULL_HANDLE) {
        if (!createPipeline(shaderModule)) return false;
    }

	// Step 4: create descriptor set
    if (!createDescriptorSet()) return false;

    // Step 5: dispatch
    if (!dispatch(inSamples, outSamples)) return false;

	// Step 6: copy output data from GPU
    if (!downloadOutputFromGPU(output, outSamples)) return false;

    outputFrames = outFrames;
    return true;
}

void VulkanUpsampler::shutdown() {
    cleanupPipeline();
    cleanupBuffers();
    cleanupVulkan();
}

bool VulkanUpsampler::initVulkan() {
    // 1. Create VkInstance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanUpsampler";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        printf("[!] Failed to create Vulkan instance\n");
        return false;
    }

    // 2. Pick first suitable GPU
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        printf("[!] No Vulkan-compatible GPUs found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    physicalDevice = devices[0]; // pick the first one

    // 3. Find compute queue family
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
        printf("[!] No compute-capable queue found\n");
        return false;
    }

    // 4. Create logical device and compute queue
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        printf("[!] Failed to create logical device\n");
        return false;
    }

    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

    // 5. Create command pool and command buffer
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        printf("[!] Failed to create command pool\n");
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to allocate command buffer\n");
        return false;
    }

    printf("[+] Vulkan initialized successfully\n");
    return true;
}

VkShaderModule VulkanUpsampler::createShaderModule(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        printf("[!] Failed to open shader file: %s\n", filename.c_str());
        return VK_NULL_HANDLE;
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        printf("[!] Failed to create shader module\n");
        return VK_NULL_HANDLE;
    }

    printf("[+] Shader module created successfully: %s\n", filename.c_str());
    return shaderModule;
}

bool VulkanUpsampler::createBuffers(uint32_t inputFrames) {
    uint32_t totalInputSamples = inputFrames * numChannels;

    // 이미 충분한 buffer가 있으면 재사용
    if (inputFrames <= maxInputFrames) {
        return true;
    }

    // 기존 buffer/memory 해제
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        inputBuffer = VK_NULL_HANDLE;
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
        outputBuffer = VK_NULL_HANDLE;
    }
    if (inputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, inputMemory, nullptr);
        inputMemory = VK_NULL_HANDLE;
    }
    if (outputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, outputMemory, nullptr);
        outputMemory = VK_NULL_HANDLE;
    }

    // 새로운 버퍼 크기
    VkDeviceSize inputSize = sizeof(float) * totalInputSamples;
    uint32_t totalOutputSamples = static_cast<uint32_t>(inputFrames * (static_cast<float>(outRate) / inRate)) * numChannels;
    VkDeviceSize outputSize = sizeof(float) * totalOutputSamples;

    // 공통 버퍼 생성 함수
    auto createBuffer = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, const char* label) -> bool {
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

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        // HOST_VISIBLE + HOST_COHERENT → 직접 맵핑해서 사용
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        bool found = false;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memRequirements.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags &
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                allocInfo.memoryTypeIndex = i;
                found = true;
                break;
            }
        }

        if (!found) {
            printf("[!] Failed to find suitable memory type for %s buffer\n", label);
            return false;
        }

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            printf("[!] Failed to allocate memory for %s buffer (size: %llu bytes)\n",
                label, static_cast<unsigned long long>(allocInfo.allocationSize));
            return false;
        }

        VkResult bindResult = vkBindBufferMemory(device, buffer, memory, 0);
        if (bindResult != VK_SUCCESS) {
            printf("[!] Failed to bind memory for %s buffer (code = %d)\n", label, bindResult);
            return false;
        }

        return true;
        };

    if (!createBuffer(inputSize, inputBuffer, inputMemory, "input")) return false;
    if (!createBuffer(outputSize, outputBuffer, outputMemory, "output")) return false;

    maxInputFrames = inputFrames;

    printf("[+] GPU buffers created: in=%.1f KB, out=%.1f KB\n",
        inputSize / 1024.0, outputSize / 1024.0);

    return true;
}

bool VulkanUpsampler::uploadInputToGPU(const float* input, uint32_t totalSamples) {
    if (inputMemory == VK_NULL_HANDLE) {
        printf("[!] Input memory not allocated\n");
        return false;
    }

    void* data = nullptr;
    VkResult result = vkMapMemory(device, inputMemory, 0, sizeof(float) * totalSamples, 0, &data);
    if (result != VK_SUCCESS) {
        printf("[!] Failed to map input memory (error %d)\n", result);
        return false;
    }

    std::memcpy(data, input, sizeof(float) * totalSamples);
    vkUnmapMemory(device, inputMemory);
    return true;
}

bool VulkanUpsampler::createPipeline(VkShaderModule shader) {
    // 1. Descriptor Set Layout (binding = 0: input, 1: output)
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

    // 2. Push Constant Range
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 2 + sizeof(float); // inSampleCount, outSampleCount, ratio

    // 3. Pipeline Layout
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

    // 4. Shader Stage
    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shader;
    shaderStage.pName = "main";

    // 5. Compute Pipeline
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
    if (descriptorSetLayout == VK_NULL_HANDLE) {
        printf("[!] DescriptorSetLayout not created\n");
        return false;
    }

    // 1. Descriptor Pool 생성
    VkDescriptorPoolSize poolSizes[1]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        printf("[!] Failed to create descriptor pool\n");
        return false;
    }

    // 2. DescriptorSet 할당
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        printf("[!] Failed to allocate descriptor set\n");
        return false;
    }

    // 3. DescriptorSet 업데이트
    VkDescriptorBufferInfo inputInfo{};
    inputInfo.buffer = inputBuffer;
    inputInfo.offset = 0;
    inputInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo outputInfo{};
    outputInfo.buffer = outputBuffer;
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

    return true;
}

bool VulkanUpsampler::dispatch(uint32_t inSamples, uint32_t outSamples) {
    if (computePipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
        printf("[!] Pipeline not ready\n");
        return false;
    }

    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        printf("[!] Failed to begin command buffer\n");
        return false;
    }

    // Bind compute pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    struct PushConstants {
        uint32_t inFrameCount;
        uint32_t outFrameCount;
        float ratio;
    };
    uint32_t outFrameCount = outSamples / 2;

    PushConstants push;
    push.inFrameCount = inSamples / 2;
    push.outFrameCount = outSamples / 2;
    push.ratio = static_cast<float>(push.outFrameCount) / push.inFrameCount;

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(push),
        &push
    );

    // Dispatch
    uint32_t totalSampleThreads = outFrameCount * 2;
    uint32_t groupCount = (totalSampleThreads + 63) / 64;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        printf("[!] Failed to record command buffer\n");
        return false;
    }

    // Submit to compute queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        printf("[!] Failed to submit compute queue\n");
        return false;
    }

    vkQueueWaitIdle(computeQueue);
    return true;
}

bool VulkanUpsampler::downloadOutputFromGPU(float* output, uint32_t totalSamples) {
    if (outputMemory == VK_NULL_HANDLE) {
        printf("[!] Output memory not allocated\n");
        return false;
    }

    void* data = nullptr;
    VkResult result = vkMapMemory(device, outputMemory, 0, sizeof(float) * totalSamples, 0, &data);
    if (result != VK_SUCCESS) {
        printf("[!] Failed to map output memory (error %d)\n", result);
        return false;
    }

    std::memcpy(output, data, sizeof(float) * totalSamples);
    vkUnmapMemory(device, outputMemory);
    return true;
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
    }
}

void VulkanUpsampler::cleanupBuffers() {
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        inputBuffer = VK_NULL_HANDLE;
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
        outputBuffer = VK_NULL_HANDLE;
    }
    if (inputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, inputMemory, nullptr);
        inputMemory = VK_NULL_HANDLE;
    }
    if (outputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, outputMemory, nullptr);
        outputMemory = VK_NULL_HANDLE;
    }
}

void VulkanUpsampler::cleanupVulkan() {
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }

    instance = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    commandPool = VK_NULL_HANDLE;
    commandBuffer = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
}
