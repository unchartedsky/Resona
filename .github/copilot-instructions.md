# Copilot Instructions

## Project Guidelines
- User prefers a workflow where they manually handle rollbacks; after rollback they want root-cause investigation rather than automatic rollback actions.
- User expects future cleanup of leftover resize-related code that remains from earlier batch-size adjustment logic.
- User considers the current `poll()` method likely unused in the main execution path and a candidate for later cleanup rather than immediate removal.