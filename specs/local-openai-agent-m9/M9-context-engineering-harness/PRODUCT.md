# M9: Context Engineering & Harness Architecture

## Summary

M9 improves Warp's local OpenAI-compatible agent by making context explicit, stable, inspectable, and easier to test across plan, implementation, follow-up, branch, and undo flows. The milestone takes a gradual path: add a first-class environment context block, reorganize prompt content for cache-friendly stable prefixes, and expand the local harness so multi-turn agent behavior can be verified before users find regressions.

## Problem

M8 closed the core plan-to-implementation loop, but it also exposed that prompt and model-interaction regressions are hard to catch with unit tests alone. The agent needs a more deliberate context architecture so it behaves more like mature coding agents: it should consistently know where it is running, what environment constraints apply, which mode it is in, and what state transition is expected next.

The user has explicitly asked to continue improving local-agent ergonomics and reliability. M9 is the product layer for that work.

## Research Baseline

- OpenAI prompt caching works best when prompts share exact static prefixes; OpenAI recommends placing static instructions and examples first and variable user-specific information at the end. OpenAI's API also reports `cached_tokens`, and caching applies automatically for supported models. See https://developers.openai.com/api/docs/guides/prompt-caching.
- Anthropic supports prompt caching over tools, system, and messages up to cache breakpoints, with automatic and explicit `cache_control` mechanisms. See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching.
- Claude Code loads persistent project/user memory such as `CLAUDE.md` at session start, treats those files as context rather than enforcement, and distinguishes behavior-shaping instructions from hard client-side settings. See https://docs.anthropic.com/en/docs/claude-code/memory.
- Cursor background agents make the execution environment explicit through repository-backed setup such as `.cursor/environment.json`, install/start commands, terminals, isolated machines, and visible handoff state. See https://docs.cursor.com/background-agent.
- Current Warp local-agent behavior already injects system prompts, read-only context, project rules, file attachments, directory context, git context, current time, local tools, plan mode, approved-plan implementation prompts, file-write tools, shell execution, and M8 plan approval state. M9 should reorganize and test these surfaces rather than replace them wholesale.

## Goals

1. Make the local agent's environment context explicit and consistently available to the model.
2. Separate stable prompt layers from dynamic per-turn context so cache-friendly prefixes are possible.
3. Keep prompt changes trace-verifiable: actual provider request bodies must show the intended layer order and dynamic context.
4. Expand the harness from single-boundary checks into multi-turn state-machine scenarios that cover plan, implementation, follow-up, branch, and undo-like transitions.
5. Preserve M8 safety invariants: plan mode does not execute changes; approved plans do not bypass tool approvals; file writes and shell commands remain governed by existing policy layers.
6. Ship gradually in small PRs rather than rewriting the whole prompt system.

## Non-Goals

1. Do not replace the local OpenAI-compatible provider stack or migrate to a different API surface as part of M9.
2. Do not build a full long-term memory system.
3. Do not autoexecute file writes or expand shell autoexecute beyond the approved M8 allowlist specs.
4. Do not change MCP, web, subagent, or file-write safety semantics unless a later M9 TECH explicitly gates that work.
5. Do not make prompt caching a hard dependency. M9 should improve prefix stability even when the upstream OpenAI-compatible proxy does not expose every cache-control feature.
6. Do not expose raw environment variables, secrets, API keys, full shell startup files, or unbounded git/status output to the model.
7. Do not rely on model compliance for safety. Enforcement remains in client-side tools, policies, and state machines.

## Behavior

1. Every normal local-agent request includes an environment context section when the underlying data is available.

2. The environment context section is clearly labeled and concise. It includes safe, bounded facts such as:
   - Operating system family and version/distribution when known.
   - CPU architecture when known.
   - Active shell family when known.
   - Current working directory.
   - Workspace root or writable root when known.
   - Git branch and head commit when known.
   - Whether the terminal session is local or SSH/remote.
   - Whether relevant local-agent feature flags are enabled at a high level, without leaking internal rollout metadata that is not useful to the model.

3. The environment context section never includes:
   - Full process environments.
   - Secret values.
   - API keys or provider credentials.
   - Raw shell rc files.
   - Full `git status` output.
   - Unbounded filesystem listings.
   - Server configuration or internal transport credentials.

4. Missing environment fields are omitted or rendered as explicit `unknown` values. The model should not receive misleading placeholder paths or fake defaults.

5. Remote/SSH sessions are represented accurately. If local file tools cannot safely operate on the remote filesystem, the context says so in user-meaningful language and the tool policy continues to enforce it.

6. Plan mode receives enough environment context to write correct paths and command plans, but plan mode still cannot execute shell commands, write files, call MCP tools, update todos, or make changes.

7. Approved-plan implementation receives the same environment context, plus the approved plan and normal-mode tools. It should continue through approved steps without asking for intermediate confirmation, while tool approvals remain enforced by the tool layer.

8. Follow-up turns preserve the relevant prior action results and current environment context without replaying stale dynamic facts as if they were current.

9. The model can distinguish static instructions from dynamic context. Static instructions should remain stable across requests unless the product behavior changes; dynamic context should appear later and be easier to inspect in traces.

10. Prompt layer order is deterministic. A reviewer should be able to inspect a request body and see the same high-level order every time:
    - Stable base instructions.
    - Stable mode/tool instructions.
    - Stable safety constraints.
    - Stable tool catalog/schema.
    - Dynamic environment context.
    - Dynamic project/file/rule context.
    - Conversation history and current user/action-result input.

11. The prompt structure is cache-friendly for providers that support prefix caching. Static layers should not include timestamps, cwd, git branch, active file names, todo snapshots, or other per-turn values.

12. If upstream cache metadata is available, Warp records enough usage information to understand whether cached prompt tokens were used. If the OpenAI-compatible proxy does not expose cache metadata, Warp still preserves stable prompt ordering and records that cache telemetry was unavailable.

13. The product behavior does not depend on any single provider's caching feature. OpenAI automatic prefix caching, Anthropic cache breakpoints, and OpenAI-compatible proxies may differ; Warp should treat provider cache support as an optimization, not a correctness requirement.

14. The local harness can simulate multi-turn plan/implementation flows without a live model. It can assert emitted provider request shape, tool catalogs, tool results, and state transitions.

15. The harness covers plan approval state transitions: entering plan mode, calling `exit_plan_mode`, showing pending approval, approving, implementing, rejecting/revising, and returning to normal follow-up.

16. The harness covers action-result resume flows: shell command output, MCP result, sync file-write result, and follow-up provider requests must route to the correct tool-use or finalize mode.

17. The harness covers branch-like flows where conversation state, git context, cwd, and file changes can differ between turns. The model should not receive stale branch/cwd facts after the context changes.

18. The harness covers undo/recovery-like flows at the product level: when a prior action is rejected, cancelled, failed, or reverted, the next request should include the correct action result and should not present the failed action as successful.

19. The harness exposes request-body trace assertions for prompt-sensitive changes. A PR that changes prompts, tool descriptions, tool catalogs, environment context, or provider request assembly must include request-body evidence or a clear reason trace is not applicable.

20. Users should experience fewer "please run this yourself" or "I cannot tell where to write" failures. The agent should more often choose the correct tool and path on the first attempt because the environment and mode are explicit.

21. Users should not see new UI surfaces solely because of M9. Any visible change in M9.1/M9.2 must be separately specified; this PRODUCT focuses on agent behavior, request assembly, and verification.

22. Existing M1-M8 user-facing behavior must continue to pass: plan approval UI, approved-plan implementation, sync file-write success cards, expand-to-view file changes, shell action-result resume, safe-command autoexecute, and Stable/Dev channel separation.

## Acceptance Criteria

1. Provider request traces show a labeled environment context block in normal mode.
2. Provider request traces show a labeled environment context block in plan mode.
3. The environment context includes cwd, OS, shell, git branch/head when those values are available.
4. The environment context omits secrets, raw environment variables, and unbounded git/filesystem output.
5. Static prompt content no longer changes when only cwd, branch, current time, todo snapshot, or user query changes.
6. Dynamic context appears after stable instructions in the request body.
7. M8.1 plan mode still blocks direct shell/file/MCP/todo mutation.
8. Approved-plan implementation still advertises normal-mode file/shell tools.
9. Shell action-result resume still re-enters ToolUse when more tools are needed.
10. Sync file-write results still render successful cards and expandable content.
11. Existing local-agent unit suites continue to pass.
12. New harness tests can assert request-layer ordering without a live upstream model.
13. New harness tests cover plan-to-approve-to-implement.
14. New harness tests cover shell result resume.
15. New harness tests cover local file-write result handling.
16. New harness tests cover follow-up after failed/cancelled/rejected action.
17. Trace evidence accompanies prompt/request assembly PRs.
18. Cache metadata is logged when available and gracefully absent when unsupported by the OpenAI-compatible proxy.
19. Dev installed-app e2e confirms the model receives the environment context and still completes the M8 plan/implement flow.
20. No Stable promotion occurs until the user explicitly confirms Dev has been stable enough.

## Safety Risks

1. **Environment information leakage:** cwd, git branch, OS, and shell are useful, but full env vars and secrets are not. The context block must be allowlist-based.
2. **Cache pollution:** putting dynamic data in the static prefix reduces cache hits and can make traces harder to reason about.
3. **Provider incompatibility:** OpenAI, Anthropic, and OpenAI-compatible proxies expose different cache controls and usage metadata. M9 must not require a feature that the configured provider does not support.
4. **State-machine regressions:** prompt reordering can accidentally break plan mode, tool-result pairing, or action-result resume. Harness coverage must lead implementation.
5. **False confidence from unit tests:** tests that only inspect helpers can miss actual provider request shape. Prompt-sensitive PRs need trace or request-body evidence.
6. **Over-broad context:** adding too much context can increase latency, costs, and model confusion. Environment context must be bounded and purposeful.

## V0 Scope

1. M9.1: Base layered prompt structure. Separate stable base instructions, mode instructions, safety instructions, tool descriptions, dynamic environment context, and conversation/history input.
2. M9.2: Environment context block. Add bounded OS/shell/cwd/git/session context and request-body trace tests.
3. M9.3: Cache-friendly ordering and cache telemetry. Move dynamic values out of the static prefix; record cached-token metadata when the provider returns it.
4. M9.4: Harness state-machine expansion. Add multi-turn request-shape and state-transition tests for plan/implementation/follow-up/action-result flows.
5. M9.5: Documentation and rollout. Capture prompt/request invariants, trace process, and Dev e2e checklist.

## V1+ Deferred Work

1. Full long-term memory or repository-learning system.
2. Cross-provider prompt-cache abstraction with Anthropic explicit cache breakpoints.
3. User-editable prompt layer UI.
4. Per-project environment setup files similar to Cursor `.cursor/environment.json`.
5. Large-scale prompt template migration or code generation framework.
6. Automated native UI e2e for every model-interaction path.

## PR Split

The expected implementation should be split into approximately five PRs:

1. **M9.1 layered prompt base:** introduce request-layer boundaries and tests that prove ordering without changing user-visible behavior.
2. **M9.2 environment context:** add bounded environment context in normal and plan mode, with request-body trace evidence.
3. **M9.3 cache-friendly reordering and telemetry:** keep dynamic values out of static prefixes, log cached-token metadata when available, and document provider fallback.
4. **M9.4 harness state machine:** expand harness scenarios for plan/implementation/follow-up/branch/undo-like flows.
5. **M9.5 docs and rollout:** update local-agent docs/process notes, install Dev, and complete user e2e gates.

Split further if a PR touches provider transport, tool policy, UI state, or execution policy beyond the planned layer.

## Open Questions for TECH.md

1. Which local request type should own the environment context block: system message, synthetic user context, or a distinct internal layer rendered into the provider request?
2. How should cache usage metadata be normalized across Chat Completions, Responses-compatible providers, and OpenAI-compatible proxies that omit `cached_tokens`?
3. What is the minimal harness abstraction that can assert provider request shape without duplicating too much production code?
4. Which branch/undo flows are real product flows today versus harness-only future placeholders?
5. Should M9.2 include architecture/cwd context for SSH sessions if local file writes remain unavailable there?
