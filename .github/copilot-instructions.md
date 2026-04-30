# Copilot Instructions

## Base Behavioral Guidelines
Apply these repository-agnostic guidelines first, then apply project-specific extensions.

### Think Before Coding
- State assumptions explicitly; if uncertain, ask for clarification.
- Surface multiple valid interpretations instead of choosing silently.
- Call out simpler approaches when they exist.
- Stop and ask when requirements are ambiguous.

### Simplicity First
- Implement only what is requested.
- Avoid speculative abstractions and unnecessary configurability.
- Prefer straightforward solutions over complex designs.

### Surgical Changes
- Change only what is required for the current task.
- Avoid unrelated refactors, formatting, or cleanup.
- Match existing style unless a structural change is requested.
- Remove only code made unused by your own edits.

### Goal-Driven Execution
- Define success criteria before implementation.
- For multi-step work, use a short plan with verification points.
- Verify behavior after changes (build/tests/repro steps as applicable).

## Project-Specific Extensions

### Project Guidelines
- Target Windows, Linux, Android, iOS, and macOS.
- Prefer a workflow where the user manually handles rollbacks; after rollback, perform root-cause investigation rather than initiating automatic rollback actions.
- Evolve architecture: treat VulkanUpsampler as one renderer path; design the system to support composing AI models and multiple renderer backends into the render pipeline and to progress the upsampler toward a sound rendering engine.
- Add a simple UI to display input/output source information and allow users to configure filter combinations and settings.
- Ensure pressure/target logic can be explicitly reinitialized or overridden and document the behavior and configuration options.

## Documentation Requirements
- Include explicit checks for Vulkan version and miniaudio version in README documentation.

## Maintenance
- Remove completed, already-applied items from Copilot instruction files to keep them current.
