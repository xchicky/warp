# LocalAgentTodoWrite Promotion Plan

## Summary

Promote conversation-scoped todo list updates for the local OpenAI-compatible agent. This is low risk because it mutates only the active conversation task list and reuses existing todo UI/history messages.

Recommended wave: 1.

## Current Status

- Implemented in M4.1 behind `FeatureFlag::LocalAgentTodoWrite`, default off.
- Exposes `todo_write` with a full-list replacement contract.
- Validates duplicate IDs, empty fields, invalid statuses, item/string limits, and multiple `in_progress` items.
- Emits existing `api::Message::UpdateTodos` operations for UI/history replay.
- Participates in the sequential/mutating tool pool.

## Known Limitations

- Todo state is conversation-scoped and intentionally not persisted across conversations.
- There is no standalone read-only todo tool; plan mode injects a bounded todo snapshot instead.
- The full-list contract is robust for replay but less efficient than partial updates.

## Dogfood Gate

1. Confirm local-agent tests pass on current `master`.
2. Manually verify a local provider can create, update, complete, and clear todos in one conversation.
3. Verify replay from history does not retain stale completed todos after full-list replacement.
4. Confirm no todo content is sent to product telemetry.

## Preview Gate

1. Dogfood for one cadence cycle with no UI replay regressions.
2. Verify plan mode todo snapshot still works when both flags are enabled.
3. Release-note copy states that todos are conversation-local.

## Stable Gate

1. Preview for one cadence cycle with no data-loss or stale-UI issue.
2. Confirm todo UI behavior matches the official agent path closely enough for support docs.

## Rollback

Demote `LocalAgentTodoWrite`. Existing todo messages in prior conversations may still render, but the local agent should stop advertising or executing `todo_write`.

## Recommended PR Order

Promote after cost telemetry and before plan mode, because plan mode benefits from the read-only todo snapshot.
