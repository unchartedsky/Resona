# Resona

Resona is a GPU-accelerated audio resampling and sound modeling engine, built for maximizing audio playback potential across platforms.

## 🔊 Current Audio Pipeline (Windows)

Spotify → VB-CABLE → Resona → DAC

- **Spotify**: Audio source  
- **VB-CABLE**: Virtual audio device for routing output  
- **Resona**: GPU-based audio upsampler (Vulkan)  
- **DAC**: Final audio output device

### Third-party dependencies

- [miniaudio](https://github.com/mackron/miniaudio) — Not included.  
  Please download `miniaudio.h` manually and place it in the root directory.
