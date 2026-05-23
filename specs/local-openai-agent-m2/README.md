# Local OpenAI-Compatible Agent: M2 MCP Server Passthrough

## Summary

M2 lets the local OpenAI-compatible agent call tools exposed by user-configured MCP servers. The milestone should reuse Warp's existing MCP settings, file discovery, server lifecycle, permission, and `CallMCPTool` execution paths rather than adding a second MCP implementation inside `app/src/ai/agent/local`.

## Recommended Split

MCP touches process lifecycle, tool schema conversion, user approval, local provider message history, and reconnect/error handling. Keep M2 split into four sub-milestones:

1. [M2.1-client-lifecycle.md](M2.1-client-lifecycle.md): feature flag, configuration source, active server discovery, and stdio lifecycle reuse.
2. [M2.2-tool-discovery.md](M2.2-tool-discovery.md): MCP tool catalog to OpenAI function tools, namespacing, and schema filtering.
3. [M2.3-invocation-passthrough.md](M2.3-invocation-passthrough.md): user-approved local MCP tool invocation and result round-trip back to the provider.
4. [M2.4-errors-timeout-reconnect.md](M2.4-errors-timeout-reconnect.md): bounded errors, timeout, reconnect, stale catalog, and regression hardening.

Each sub-milestone should be independently testable and revertable. M2.2 depends on M2.1; M2.3 depends on M2.2; M2.4 depends on M2.3.

## Configuration Recommendation

Use existing Warp MCP configuration and discovery as the source of truth. Do not add a local-agent-specific MCP config file in M2.

Recommended behavior:

- User-visible setup remains Settings -> MCP Servers and file-based MCP discovery already handled by Warp.
- Project-scoped config should be the primary repo-local authoring path because it matches common MCP usage and keeps project tools with the project.
- Global user configs remain supported through existing Warp discovery and settings state, but M2 should consume the active server set from Warp managers rather than reading global files directly from the local-agent module.

Rationale:

- The repo already has `rmcp`, `TemplatableMCPServerManager`, `FileBasedMCPManager`, `ReconnectingPeer`, MCP settings UI, and action/result proto conversions.
- Reusing the existing source of truth avoids divergent config semantics between Oz, third-party harnesses, and local OpenAI-compatible agents.
- Existing file discovery already recognizes Warp, Claude, Codex, and generic agent MCP config locations.

## Approval Recommendation

Default to visible approval for every MCP tool call in M2. Do not introduce first-use auto-approval or a new trust cache.

Reasons:

- MCP servers are user-configured external processes and individual tool calls may read, write, or exfiltrate data depending on the server.
- Cursor and Claude-style agent experiences keep MCP tool calls visible and let users explicitly opt into broader trust.
- Warp already has MCP permission concepts; M2 should route through the existing approval path instead of inventing local-agent-only policy.

If existing Warp MCP settings already contain an explicit user-controlled allow/deny decision, the implementation may reuse that decision only if it goes through the same `CallMCPTool` permission path used by non-local agents. Newly discovered or newly enabled MCP tools must not auto-run just because `LocalAgentMcp` is enabled.

## External Reference Points

- MCP transport spec: `stdio` is a standard transport and each MCP server usually runs as its own process. Reference: https://modelcontextprotocol.io/specification/draft/basic/transports
- Claude Code supports project and user MCP scopes, project `.mcp.json` files with `mcpServers`, and approval before using project-scoped servers. Reference: https://docs.claude.com/en/docs/claude-code/mcp
- Cursor asks for approval before using MCP tools by default and has opt-in auto-run. Reference: https://docs.cursor.com/en/context/mcp
- Codex-style MCP configs are already normalized by Warp from `[mcp_servers.<name>]` TOML sections in `~/.codex/config.toml`; see `app/src/ai/mcp/parsing.rs`.

## Non-Goals For M2

- Do not add SSE or streamable HTTP transport support for local-agent MCP passthrough in the first M2 implementation.
- Do not add MCP resources as first-class OpenAI tools unless a sub-milestone explicitly expands scope.
- Do not add TodoWrite, skills, subagents, codebase index, web search, or token telemetry.
- Do not change remote/Oz MCP behavior except where shared code needs test-only helpers or bug fixes.
