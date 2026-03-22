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
- richer interpolation / synthesis strategies
- GPU-driven sound generation or transformation
- more flexible scheduling and buffering policies

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
