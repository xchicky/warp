# LocalAgentCodebaseIndex Promotion Plan

## Summary

Promote local-only codebase search for the local OpenAI-compatible agent. V0 uses a process-local in-memory lexical index and avoids cloud embeddings, vector databases, or persisted artifacts.

Recommended wave: 1.

## Current Status

- Implemented in M5.2 behind `FeatureFlag::LocalAgentCodebaseIndex`, default off.
- Exposes V0 `search_codebase`; `find_references` is a follow-up.
- Uses a standalone local lexical index under `app/src/ai/agent/local`.
- Enforces canonical workspace containment and symlink escape denial.
- Skips generated, large, binary, and non-UTF-8 files.
- Bounds indexing/output and surfaces stale status to the provider.
- Plan mode allows `search_codebase` as read-only.

## Known Limitations

- Cache is process-local and cleared on process exit.
- Search is lexical, not semantic.
- No persistent incremental index and no `find_references` in V0.

## Dogfood Gate

1. Confirm local-agent and `codebase_index` tests pass on current `master`.
2. Repeat red-line grep for cloud embedding/store/vector paths in local-agent code.
3. Manually verify search in a medium repository and stale status after file edits.
4. Confirm no repo contents are written outside process memory or sent to product telemetry.

## Preview Gate

1. Dogfood for one cadence cycle with acceptable latency on representative repositories.
2. Define user-facing copy for stale/partial index states.
3. Confirm generated/large/binary skip metadata is bounded and not noisy.

## Stable Gate

1. Preview for one cadence cycle with no source-leak or path-containment issue.
2. Decide whether persistence is still a follow-up or required before stable. Stable can proceed without persistence if V0 latency is acceptable.

## Rollback

Demote `LocalAgentCodebaseIndex`. Process-local caches disappear on process exit; no disk cleanup is required for V0.

## Recommended PR Order

Promote late in wave 1, before plan mode and subagent, because those features benefit from reliable read-only repository context.
