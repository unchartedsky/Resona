# Resona

Resona is a GPU-accelerated audio resampling and sound modeling engine, built for maximizing audio playback potential across platforms.

### Third-party dependencies

- [miniaudio](https://github.com/mackron/miniaudio) — Not included.  
  Please download `miniaudio.h` manually and place it in the root directory.  
  - ✅ Verified with: **miniaudio v0.11.25**  
  - 🔍 Version check:  
    - Open `miniaudio.h` and check the header comment near the top of the file  
    - Expected format example: `miniaudio - v0.11.25 - 2026-03-04`

- **Vulkan SDK** — Required for compute shader execution.  
  - ✅ Verified with: **Vulkan SDK 1.4.341.1** (Tested on AMD Radeon RX9070)  
  - ✅ Minimum required: **Vulkan 1.2**  
  - 🔗 Download: https://vulkan.lunarg.com/  
  - 🔍 Version check:  
    - Run `vulkaninfo`  
    - Or confirm the installed SDK path/version under `C:\VulkanSDK\<version>\`
  > Make sure your GPU driver supports Vulkan 1.2 or later.  
  > You can check your system support using `vulkaninfo`.

## 🧪 Shader Compilation

Resona uses **GLSL compute shaders** compiled to **SPIR-V** using `glslangValidator`.

### ✳ Required Tool

- `glslangValidator` — Included in the Vulkan SDK  
  - Windows: `C:\VulkanSDK\<version>\Bin\glslangValidator.exe`

### 🛠 Compile Example

```bash
glslangValidator -V nearest.comp -o nearest.spv
```

## 🔊 Current Audio Pipeline (Windows)

Application (e.g. Spotify) → VB-CABLE → Resona → DAC

- **Application (Spotify, etc.)**:  
  Any application that can select **VB-CABLE** as the output device via Windows Volume Mixer.
- **VB-CABLE**: Virtual audio device for routing output  
- **Resona**: GPU-based audio upsampler (Vulkan)  
- **DAC**: Final audio output device  
  - Must support **384kHz** output.  
  - ✅ Tested devices: **Questyle M12i**, **iFi Uno**

## 🪪 License

This project is licensed under the [MIT No Attribution (MIT-0)](LICENSE) license.  
You may use, modify, and distribute it freely without attribution.
