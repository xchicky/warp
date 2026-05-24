# Local OpenAI-Compatible Agent: M5 Codebase Index, Web, and Subagents

## Summary

M5 closes the local OpenAI-compatible agent parity roadmap with three larger capabilities:

1. Web search and fetch, so the local agent can gather current public information through explicit tools.
2. A local codebase index/search path, so the agent can answer repository questions without repeatedly reading broad file sets.
3. Local subagents, so the agent can delegate bounded tasks to isolated child contexts and merge results back into the main conversation.

These features are intentionally independent. They all add new context sources, but each has a different risk profile: outbound network access, local persistence/indexing, and multi-agent control flow. Keep the implementation sequence conservative and reviewable.

## Recommended Split

Split M5 into three sub-milestones:

1. [M5.1-web-search-fetch.md](M5.1-web-search-fetch.md): feature flag, `web_search` and `web_fetch` schemas, SearXNG-backed search, HTTP fetch safety, output conversion, and existing web UI reuse.
2. [M5.2-codebase-index.md](M5.2-codebase-index.md): feature flag, local-only codebase index/search adapter, bounded tool output, persistence/staleness policy, and existing SearchCodebase UI/action reuse.
3. [M5.3-subagent.md](M5.3-subagent.md): feature flag, local `spawn_subagent` tool, isolated child context, same approval/policy gates, bounded results, and existing subagent UI/action reuse where practical.

Recommended implementation order is M5.1 -> M5.2 -> M5.3. Web is first because it is independent and forces the outbound privacy/SSRF policy to be reviewed early. Codebase index is second because it adds local persistence and can later improve subagent usefulness. Subagent is last because it is the highest control-flow risk and should inherit the final local tool set.

## Shared Constraints

- All capabilities must be guarded by default-off feature flags:
  - `FeatureFlag::LocalAgentWeb`
  - `FeatureFlag::LocalAgentCodebaseIndex`
  - `FeatureFlag::LocalAgentSubagent`
- Specs must land in source control before implementation PRs start. Agent1 should branch from post-spec `origin/master`.
- Do not change remote/Oz provider behavior except where shared action/result/UI primitives need narrow reuse.
- Do not send local content, web queries, fetched URLs, fetched page content, index contents, or subagent task text to product telemetry or Warp billing.
- Continue routing local tool-result resume/finalize rounds to the local provider. Do not fall back to remote/Oz routes.
- Preserve existing local-agent tests and add focused tests for each sub-milestone.
- Existing mutating/sequential tool pool remains:
  - `apply_file_diff`
  - `write_file`
  - `edit_file`
  - `run_shell_command`
  - `CallMcpTool`
  - `todo_write`

## Design Recommendations

### Web

Start with a user-configured SearXNG endpoint for search and simple HTTP GET for fetch. This avoids adding a third-party API key or browser automation dependency. Use existing `WebSearchStatus`, `WebFetchStatus`, `WebSearchView`, and `WebFetchView` for visible status where practical.

`web_fetch` should be a single-page fetch tool, not a crawler. Do not add headless browsing, JavaScript execution, login/session handling, form submission, or multi-hop crawling in M5.

### Codebase Index

Warp already has `SearchCodebase` actions/UI and a full-source-code embedding index manager. M5 local-agent codebase index must stay local. Do not use the existing cloud embedding store/vector DB unless a local store/client implementation is explicitly added and gated.

Start with a local lexical/symbol/ripgrep-backed index or adapter that can be persisted locally and refreshed incrementally. Semantic embeddings can be a future enhancement if they run locally and keep source content on the user's machine.

### Subagents

Warp already has orchestration actions, child agent views, and `RunAgents`/`StartAgent` primitives. M5 should reuse those surfaces where practical, but the local OpenAI-compatible child agent should run as a single-process task/actor, not an OS process fork. A child context must be isolated from the parent conversation history and inherit the same feature flags, plan-mode restrictions, and approval gates.

Prefer a synchronous, bounded `spawn_subagent` V0. Asynchronous multi-child orchestration can be a follow-up if the synchronous result path proves insufficient.

## External Reference Points

- SearXNG provides a self-hostable metasearch endpoint with JSON output and no required third-party search key.
- Claude Code and similar tools keep subagent contexts isolated and return summarized results to the parent agent.
- Existing Warp code already includes web status messages, codebase search action/result types, and orchestration UI primitives that should be reused instead of duplicated.

## Non-Goals For M5

- Do not add headless browser automation, JavaScript rendering, authenticated browsing, cookie jar reuse, or form submission.
- Do not add third-party hosted search as the default path.
- Do not upload source code, index artifacts, web content, or subagent transcripts to Warp services.
- Do not add cloud vector DB or remote embedding dependencies for the local codebase index.
- Do not add long-running autonomous background subagent swarms, cross-process worker pools, or hidden child agents.
- Do not promote feature flags to dogfood/preview/stable. Promotion happens after M5.
