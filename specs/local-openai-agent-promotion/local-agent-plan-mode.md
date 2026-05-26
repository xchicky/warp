# LocalAgentPlanMode Promotion Plan

## Summary

Promote read-only plan mode for the local OpenAI-compatible agent. This flag changes agent control flow rather than adding a single tool, so it should move after lower-risk tools have dogfood signal.

Recommended wave: 3.

## Current Status

- Implemented in M4.2 behind `FeatureFlag::LocalAgentPlanMode`, default off.
- Activates from `/plan` user query mode when the flag is enabled.
- Enforces plan mode at both tool advertisement and execution layers.
- Allows read-only local tools, `search_codebase`, `web_search`, `web_fetch`, `suggest_shell_command`, read-only subagents, and `exit_plan_mode`.
- Denies file writes, shell execution, MCP, `todo_write`, and hidden direct forbidden tool names.
- Keeps the local conversation id and local provider route.
- Injects bounded read-only todo snapshot context when available.
- Uses round-cap fallback if the provider does not call `exit_plan_mode`.

## Known Limitations

- Provider compliance affects UX; executor hard limits preserve safety but can still yield awkward loops.
- Plan approval UI/flow depends on existing query mode surfaces.
- `todo_write` is not available inside plan mode.

## Dogfood Gate

1. Confirm local-agent plan-mode tests pass on current `master`.
2. Manually verify `/plan` cannot mutate files, todos, shell, or MCP even with those flags enabled.
3. Manually verify `exit_plan_mode` returns a complete assistant-visible plan without executing changes.
4. Verify round-cap fallback is understandable and user-interruptible.

## Preview Gate

1. Dogfood for one cadence cycle with no prompt-injection bypass or accidental execution issue.
2. Confirm release notes clearly explain plan mode is read-only until explicit user action.
3. Verify interactions with `LocalAgentTodoWrite`, `LocalAgentCodebaseIndex`, `LocalAgentWeb`, and `LocalAgentSubagent` when those flags are enabled.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 control-flow issue.
2. Confirm support docs explain how users enter, exit, approve, modify, or cancel a plan.

## Rollback

Demote `LocalAgentPlanMode`. `/plan` should return to the pre-M4.2 local-agent behavior and must not leave a conversation stuck in plan mode.

## Recommended PR Order

Promote first in wave 3 if lower-risk read-only tools have dogfood signal. Do not promote subagent to preview before plan-mode propagation has dogfood coverage.
