# LocalAgentSubagent Promotion Plan

## Summary

Promote bounded local subagent delegation for the local OpenAI-compatible agent. This is high control-flow risk because it adds child provider rounds and child tool selection, even though V0 is synchronous and single-process.

Recommended wave: 3.

## Current Status

- Implemented in M5.3 behind `FeatureFlag::LocalAgentSubagent`, default off.
- Exposes `spawn_subagent` with bounded task, context, tools, and label.
- Runs synchronously in-process using the same local provider config as the parent.
- Does not spawn OS processes, fork, or fall back to remote/Oz.
- Uses isolated child conversation ids and does not inherit the full parent transcript.
- Validates requested tools against the parent's advertised catalog and local tool policy.
- Excludes mutating/out-of-band/MCP tools from child catalogs in V0.
- Propagates plan mode and prevents child `exit_plan_mode`.
- Caps depth, child count per batch, child provider rounds, timeout, and result size.
- Drops child reasoning deltas from parent-visible summaries.

## Known Limitations

- V0 is synchronous and supports depth 1 and one child per parent batch.
- Child token/cost telemetry is not merged in V0.
- Child todo state is not separately surfaced in UI.
- Child cannot perform mutating or approval-required tools in V0.

## Dogfood Gate

1. Confirm local-agent subagent tests pass on current `master`.
2. Repeat red-line grep for OS process/fork/remote fallback terms in local-agent code.
3. Manually verify read-only child delegation with read/search/codebase/web tools.
4. Verify parent plan mode propagates and child cannot call `exit_plan_mode`.
5. Verify child cannot request hidden/mutating/MCP/shell/todo tools even when parent flags are enabled.

## Preview Gate

1. Dogfood for one cadence cycle with no permission inheritance or context-leak issue.
2. Confirm child output is clearly labeled and bounded in UI/provider context.
3. Decide whether child cost telemetry omission remains acceptable for preview.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 control-flow issue.
2. Confirm async multi-child, child mutating approval, and child cost telemetry remain follow-ups rather than stable blockers.

## Rollback

Demote `LocalAgentSubagent`. The parent local agent stops advertising `spawn_subagent`; no child state migration is required for V0.

## Recommended PR Order

Promote after plan mode dogfood and after read-only context tools have dogfood signal. Do not bundle with shell execution promotion.
