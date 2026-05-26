# LocalAgentMcp Promotion Plan

## Summary

Promote MCP lifecycle, discovery, invocation, and hardening for the local OpenAI-compatible agent. This flag connects the local provider path to Warp's existing MCP server context and approval model.

Recommended wave: 1, last within the wave.

## Current Status

- Implemented across M2.1-M2.4 behind `FeatureFlag::LocalAgentMcp`, default off.
- Reuses Warp MCP context; no independent local MCP server runner/parser.
- Discovers MCP tools with namespacing, schema conversion, collision defense, and secret redaction.
- Invokes MCP tools out of band through `CallMcpTool` with visible approval.
- Uses original MCP tool names for server invocation.
- Handles timeout/reconnect/errors with bounded provider-visible results.
- MCP tools are sequential with other mutating/out-of-band local tools.

## Known Limitations

- Relies on user-configured MCP servers and their reliability.
- Tool catalog is refreshed each round; a TTL cache remains a possible follow-up.
- Plan mode denies all MCP tools.

## Dogfood Gate

1. Confirm local-agent and MCP-focused tests pass on current `master`.
2. Manually verify at least two common MCP servers, including one tool requiring approval.
3. Verify stale server/tool identity after approval fails closed.
4. Confirm secret redaction in tool names, descriptions, and error summaries.

## Preview Gate

1. Dogfood for one cadence cycle with no approval-bypass or stale-identity issue.
2. Decide whether list-tools TTL caching is required before preview.
3. Document that MCP behavior depends on existing user MCP server configuration.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 MCP incident.
2. Confirm reconnect/error behavior is understandable for disconnected servers.

## Rollback

Demote `LocalAgentMcp`. MCP servers remain configured in Warp, but the local OpenAI-compatible agent no longer advertises or invokes MCP tools.

## Recommended PR Order

Promote last in wave 1 because it depends on external local servers and approvals, even though it reuses established MCP infrastructure.
