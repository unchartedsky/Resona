# Copilot Instructions

## Project Guidelines
- Target Windows, Linux, Android, iOS, and macOS.
- Prefer a workflow where the user manually handles rollbacks; after rollback, perform root-cause investigation rather than initiating automatic rollback actions.
- Evolve architecture: treat VulkanUpsampler as one renderer path; design the system to support composing AI models and multiple renderer backends into the render pipeline and to progress the upsampler toward a sound rendering engine.
- Add a simple UI to display input/output source information and allow users to configure filter combinations and settings.
- Document the forward-looking architecture and renderer composition strategy in AGENTS.md.

## Documentation Requirements
- Include explicit checks for Vulkan version and miniaudio version in README documentation.

## Maintenance
- Remove completed, already-applied items from Copilot instruction files to keep them current.