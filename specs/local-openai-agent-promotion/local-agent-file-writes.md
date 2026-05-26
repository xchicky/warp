# LocalAgentFileWrites Promotion Plan

## Summary

Promote local file mutation tools for the local OpenAI-compatible agent. This flag covers `apply_file_diff`, `write_file`, and `edit_file`, all scoped to the active workspace with path-safety checks.

Recommended wave: 2.

## Current Status

- Implemented across M1.1-M1.2 behind `FeatureFlag::LocalAgentFileWrites`, default off.
- Advertises `apply_file_diff`, `write_file`, and `edit_file` only when enabled.
- Enforces workspace containment and symlink escape denial.
- Emits existing apply-file-diff card/result shapes where practical.
- Participates in the sequential/mutating tool pool.
- Plan mode denies all file mutation even if the flag is enabled.

## Known Limitations

- Mutations affect the user's working tree.
- `write_file` full replacement requires explicit overwrite for existing files.
- Complex merge/conflict UX remains outside this flag.

## Dogfood Gate

1. Confirm local-agent file-write tests pass on current `master`.
2. Manually verify create, overwrite, edit, and patch flows in a throwaway repository.
3. Verify path traversal and symlink escape test fixtures fail closed.
4. Verify tool cards clearly show changed files and bounded summaries.

## Preview Gate

1. Dogfood for one cadence cycle with no workspace data-loss issue.
2. Confirm users can inspect changes through existing diff/history surfaces.
3. Plan mode deny behavior is manually verified with file-write flag enabled.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 file mutation issue.
2. Confirm rollback does not leave partially enabled tool definitions.

## Rollback

Demote `LocalAgentFileWrites`. Existing file changes remain in the user's working tree, but the local agent stops advertising/executing file mutation tools.

## Recommended PR Order

Promote before shell execution and before subagent preview. File writes are risky, but they have narrower authority than shell commands.
