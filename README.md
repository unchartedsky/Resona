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

## Runtime Buffering And Drift Control

Resona currently uses a startup calibration model for output buffer control.

- On startup, the runtime captures the initial output buffer level and uses it as the adaptive baseline.
- A fixed **15%** bootstrap target is kept only as a safety fallback until startup calibration is established.
- After startup, drift handling is intended to follow observed device behavior rather than force a permanent fixed setpoint.
- Playback restarts or diagnostic flows may explicitly reinitialize the pressure/target state.

This is important because different DACs may drift in different directions over time. One device may slowly drain the output margin while another may slowly accumulate it.

## Runtime Telemetry

The live status line is intended to help diagnose underruns, drift behavior, and device-specific playback characteristics.

Example shape:

`Out:160417(15%-15238) Min:120000 GPU:2/10 Ratio:8.707962(+0.005%) Target:95% ZF:0 159s`

Field meanings:

- `Out`
  - Current output ring buffer level in frames.
  - The percentage in parentheses is the current fill ratio versus total output ring capacity.
  - The signed value in parentheses is the recent output buffer change rate.
- `Min`
  - Lowest observed output buffer level seen during the current session.
  - This is usually more important than average fill level when diagnosing audible glitches.
- `GPU`
  - Busy GPU processing slots versus total available slots.
- `Ratio`
  - Current resampling ratio.
  - The percentage in parentheses shows drift relative to the base ratio.
- `Target`
  - Current output buffer level relative to the adaptive target/baseline.
- `ZF`
  - Zero-fill event count.
  - A zero-fill event means playback requested samples that were not ready, so the remaining region was filled with silence.

When diagnosing playback quality, prioritize these signals in this order:

1. `ZF` increasing
2. `Min` trending too low
3. `Ratio` drifting persistently in one direction for a specific DAC

## Underrun And Recovery Notes

- Playback callbacks zero-fill any missing samples rather than reading invalid data.
- If the remaining output margin drops below the configured minimum threshold, the runtime marks an underrun condition.
- The main runtime loop then attempts a playback restart instead of leaving recovery to detached background logic.

This behavior is intentionally conservative and is meant to keep the audio I/O path predictable while drift tuning continues.

## 🪪 License

This project is licensed under the [MIT No Attribution (MIT-0)](LICENSE) license.  
You may use, modify, and distribute it freely without attribution.
