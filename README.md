# Resona

Resona is a GPU-accelerated audio resampling and sound modeling engine, built for maximizing audio playback potential across platforms.

### Third-party dependencies

- [miniaudio](https://github.com/mackron/miniaudio) — Not included.  
  Please download `miniaudio.h` manually and place it in the root directory.

- **Vulkan SDK** — Required for compute shader execution.  
  - ✅ Verified with: **Vulkan SDK 1.4.321**  
  - ✅ Minimum required: **Vulkan 1.2**  
  - 🔗 Download: https://vulkan.lunarg.com/  
  > Make sure your GPU driver supports Vulkan 1.2 or later.  
  > You can check your system support using `vulkaninfo`.

## 🧪 Shader Compilation

Resona uses **GLSL compute shaders** compiled to **SPIR-V** using `glslangValidator`.

### ✳ Required Tool

- `glslangValidator` — Included in the Vulkan SDK  
  - Windows: `C:\VulkanSDK\<version>\Bin\glslangValidator.exe`

### 🛠 Compile Example

```bash
glslangValidator -V linear.comp -o linear.spv
```

## 🔊 Current Audio Pipeline (Windows)

Spotify → VB-CABLE → Resona → DAC

- **Spotify**: Audio source  
- **VB-CABLE**: Virtual audio device for routing output  
- **Resona**: GPU-based audio upsampler (Vulkan)  
- **DAC**: Final audio output device
