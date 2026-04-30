# AGENTS.md

## Project Direction

Resona is evolving from a GPU upsampler into a broader sound rendering engine.

### Long-term goals
- Support **Windows, Linux, Android, iOS, and macOS**
- Keep the **input/output path robust and predictable**
- Allow the **upsampler / renderer side** to evolve more aggressively as rendering features grow

## Architecture Priorities

### 1. Input / Output stability first
The audio device layer, buffering model, restart/recovery behavior, and platform-facing I/O code should remain conservative and reliable.

This part of the system should prioritize:
- predictable behavior
- clean shutdown and restart handling
- minimal glitching / underrun risk
- platform portability
- clear synchronization rules

### 2. Renderer / upsampler evolution second
The GPU processing path can evolve toward a sound rendering engine with more advanced behavior over time.

This part of the system may eventually include:
- multiple rendering kernels
- composition of multiple renderer backends within one render pipeline
- chaining or combining AI models with renderer stages
- richer interpolation / synthesis strategies
- GPU-driven sound generation or transformation
- more flexible scheduling and buffering policies

## Current Runtime Shape

The current codebase is moving toward a thin application entry point with explicit subsystem boundaries.

### Runtime layers
- `Runtime/`
  - application orchestration
  - shared runtime hooks
  - status and session reporting
- `Audio/`
  - device lifecycle
  - callback-facing state
  - ring buffer transport
  - shared audio configuration
- `RenderPipeline/`
  - GPU worker thread
  - submission planning and scheduling policy
- `VulkanUpsampler`
  - one renderer backend implementation inside the broader rendering direction

### Near-term intent
- keep `main.cpp` as a thin entry point and high-level control handoff
- keep audio callback paths explicit and minimal
- keep renderer scheduling policy separable from device and buffer management
- treat `VulkanUpsampler` as one backend path, not the final architecture boundary

## Buffering And Drift Control Notes

- Startup calibration should capture the initial output buffer level and use it as the adaptive baseline.
- The fixed 15% bootstrap target is only a safety fallback before startup calibration is established.
- Ongoing pressure / target behavior should be driven primarily by captured startup conditions and live drift telemetry, not by a permanently enforced fixed setpoint.
- Reinitializing or overriding pressure / target state should remain an explicit runtime action so device restarts and diagnostics can reset the controller cleanly.

## Telemetry Priorities

When diagnosing playback quality, prioritize worst-case margin over average fill level.

Important runtime signals:
- current output buffer level
- minimum observed output buffer level
- zero-fill event count
- zero-fill sample count
- adaptive ratio drift relative to base ratio

These metrics should guide tuning for device-specific drift behavior and underrun recovery.

## API Design Guidance

When possible:
- keep the public API surface small
- move implementation helpers to private scope
- prefer stable high-level entry points over exposing low-level internal helpers

## Cleanup Guidance

When identifying cleanup candidates:
- remove completed, already-applied temporary code paths
- remove pilot / proof-of-concept code once it is no longer used
- prefer simplifying legacy synchronous or unused helper paths if the active architecture is async

## Practical Interpretation For Current Codebase

- Treat the **audio I/O path** as the stable foundation
- Treat the **GPU upsampler path** as the main experimentation and expansion area
- Favor correctness, recoverability, and maintainability over premature optimization in shared infrastructure
