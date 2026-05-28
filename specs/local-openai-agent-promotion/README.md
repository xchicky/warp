# Local OpenAI-Compatible Agent: Feature Flag Promotion Roadmap

## Summary

M1-M5 delivered the local OpenAI-compatible agent parity roadmap behind 11 default-off feature flags. The post-M5 goal is to promote those flags deliberately from debug-only availability into dogfood, preview, and eventually stable use without losing the safety properties established during implementation.

Promotion must be incremental. Each flag gets its own implementation PR so rollout and rollback remain narrow. This directory defines the rollout order, dependency graph, per-flag readiness criteria, and rollback strategy.

## Promotion States

- **Debug**: flag exists and is default off. The feature is available only to explicit local overrides or test configuration.
- **Dogfood**: enabled for internal dogfood users or an equivalent internal cohort. The feature remains easy to demote.
- **Preview**: available to an external opt-in cohort with user-facing release notes and support expectations.
- **Stable**: enabled by default for the relevant local-agent audience. Flag cleanup can be considered only after a separate soak period with no incidents.

Promotion PRs should use the `promote-feature` skill and must not promote more than one flag at a time.

Dogfood validation for promoted local-agent flags must use `WarpLocal.app` / `Channel::Local`, which enables `DEBUG_FLAGS`, `DOGFOOD_FLAGS`, and `PREVIEW_FLAGS`. `WarpOss.app` / `Channel::Oss` enables only `DEBUG_FLAGS`, so it is not a valid app for dogfood verification of promoted local-agent features.

For local macOS dogfood builds without the private channel config binary installed, use the local-channel bundle path: `PATH="/tmp/warp-channel-stub-bin:$PATH" WARP_BIN_NAME=warp WARP_CHANNEL=local FEATURES=gui,release_bundle ./script/macos/run --dont-open`.

## Cadence

Use one cadence cycle as the minimum feedback window between promotions that materially increase risk. For this roadmap, one cadence cycle means at least five business days or one internal release train, whichever is longer.

Within a wave, low-risk flags can be prepared in parallel as specs, but implementation PRs still land one flag at a time. If a flag produces a rollback or severity-1/2 issue, pause subsequent promotions in the same risk family until the issue is understood.

## Dependency Graph

```text
Local OpenAI-compatible base
|-- LocalAgentCostTelemetry
|-- LocalAgentImageInput
|-- LocalAgentTodoWrite -----.
|-- LocalAgentSkills --------+--> LocalAgentPlanMode --.
|-- LocalAgentCodebaseIndex -'
|-- LocalAgentMcp
|-- LocalAgentWeb ------------------------------------+--> LocalAgentSubagent
|-- LocalAgentFileWrites
`-- LocalAgentShellExecution
```

Notes:

- `LocalAgentPlanMode` has no hard runtime dependency on `TodoWrite`, `CodebaseIndex`, `Web`, or `Skills`, but its user experience is materially better once read-only planning context is available.
- `LocalAgentSubagent` inherits the parent tool catalog and policy. Promote it only after the inherited read-only tools and plan-mode policy have dogfood signal, and before enabling child access to any future mutating approval path.
- `LocalAgentMcp` depends on user-configured MCP servers. It can dogfood early, but preview should wait for common server coverage and stale-server/reconnect issue review.

## Recommended Waves

### Wave 1: Low External-Side-Effect Flags

Promote these to dogfood first:

1. [LocalAgentCostTelemetry](local-agent-cost-telemetry.md)
2. [LocalAgentTodoWrite](local-agent-todo-write.md)
3. [LocalAgentImageInput](local-agent-image-input.md)
4. [LocalAgentSkills](local-agent-skills.md)
5. [LocalAgentCodebaseIndex](local-agent-codebase-index.md)
6. [LocalAgentMcp](local-agent-mcp.md)

These features either observe local/provider state, mutate only conversation-local state, or operate on explicit user-configured local context. `LocalAgentMcp` is included in wave 1 because Warp already has an MCP permission model, but it should be sequenced last inside the wave because it depends on external local servers.

### Wave 2: Local/Network Side Effects

Promote after wave 1 dogfood signal:

1. [LocalAgentFileWrites](local-agent-file-writes.md)
2. [LocalAgentWeb](local-agent-web.md)

File writes affect the workspace, while web access performs outbound HTTP. Both have focused safety tests, but they should not be promoted until low-risk local-agent surfaces are being exercised by dogfood users.

### Wave 3: Highest Control-Flow Risk

Promote last:

1. [LocalAgentPlanMode](local-agent-plan-mode.md)
2. [LocalAgentSubagent](local-agent-subagent.md)
3. [LocalAgentShellExecution](local-agent-shell-execution.md)

Plan mode changes agent control flow, subagents add child execution contexts, and shell execution can run commands. Each should receive a dedicated dogfood window before the next high-risk flag moves.

## Global Entry Criteria

Before any flag leaves debug:

1. The flag is still default off on `master`.
2. The feature has focused local-agent tests and they pass on current `master`.
3. The PR description lists known limitations and rollback command/path.
4. No known open severity-1/2 issue exists for the feature area.
5. Product telemetry, billing, and unsafe logs do not capture local content, tool arguments, image bytes, fetched page content, skill content, code snippets, shell output, or subagent task/context.
6. The promotion PR changes only rollout plumbing for one flag unless PM explicitly approves otherwise.

## Global Rollback

Rollback should demote the single affected flag to its previous state. Do not revert unrelated local-agent code unless the feature cannot be isolated by flag state.

For dogfood and preview incidents:

1. Demote the flag with a focused PR or emergency config change, whichever the flag system supports for that state.
2. Leave code paths intact when the flag can disable behavior.
3. Add or update a regression test before re-promoting.
4. Document the incident and new gate in the affected per-flag spec if the gate changes.

Stable rollback may require both demotion and a follow-up cleanup reversal if the flag check has already been removed. Do not remove any local-agent flag until a separate stable soak period has completed.

## Non-Goals

- Do not add new local-agent capabilities in promotion PRs.
- Do not combine multiple flag promotions in one PR.
- Do not clean up or remove a flag in the same PR that promotes it.
- Do not start M6 follow-ups such as MCP TTL caching, persistent codebase indexes, `find_references`, async multi-child subagents, or additional web providers as part of this roadmap.

## Spec Index

- [LocalAgentCostTelemetry](local-agent-cost-telemetry.md)
- [LocalAgentTodoWrite](local-agent-todo-write.md)
- [LocalAgentImageInput](local-agent-image-input.md)
- [LocalAgentSkills](local-agent-skills.md)
- [LocalAgentCodebaseIndex](local-agent-codebase-index.md)
- [LocalAgentMcp](local-agent-mcp.md)
- [LocalAgentFileWrites](local-agent-file-writes.md)
- [LocalAgentWeb](local-agent-web.md)
- [LocalAgentPlanMode](local-agent-plan-mode.md)
- [LocalAgentSubagent](local-agent-subagent.md)
- [LocalAgentShellExecution](local-agent-shell-execution.md)
