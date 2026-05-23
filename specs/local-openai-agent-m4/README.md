# Local OpenAI-Compatible Agent: M4 TodoWrite, Plan Mode, and Skills

## Summary

M4 adds three agent-behavior capabilities to the local OpenAI-compatible agent:

1. A conversation-scoped `todo_write` tool that lets the local agent maintain visible task state.
2. Plan mode, a read-only mode with hard tool restrictions and explicit plan exit.
3. Local skill loading from Claude-style skill directories.

These features are related because they shape agent workflow, but they should ship as three sub-milestones. TodoWrite establishes reusable conversation task state. Plan mode builds a read-only execution policy and can optionally read current todos. Skill loading adds local prompt context and slash-triggered skill content without making skills executable code.

## Recommended Split

Keep M4 split into three sub-milestones:

1. [M4.1-todowrite.md](M4.1-todowrite.md): feature flag, `todo_write` schema, conversation-scoped todo state, existing todo UI/history reuse, and scheduler integration.
2. [M4.2-plan-mode.md](M4.2-plan-mode.md): feature flag, plan-mode state machine, read-only tool policy, `exit_plan_mode`, and conversation resume behavior.
3. [M4.3-skills.md](M4.3-skills.md): feature flag, local Claude-style skill discovery, metadata injection, slash-triggered full skill loading, and path containment.

M4.2 depends on M4.1 if the implementation exposes current todo state or TodoWrite read-only operations in plan mode. M4.3 is mostly independent and can start after M4.1 or M4.2, but keep the implementation sequence M4.1 -> M4.2 -> M4.3 unless review bandwidth favors parallel branches.

## Shared Constraints

- All capabilities must be guarded by default-off feature flags:
  - `FeatureFlag::LocalAgentTodoWrite`
  - `FeatureFlag::LocalAgentPlanMode`
  - `FeatureFlag::LocalAgentSkills`
- Do not change remote/Oz provider behavior except where shared conversation, todo, or skill primitives need narrow reuse.
- Do not add M5 capabilities such as codebase index, subagents, web search, or fetch.
- Do not rely on prompt text alone for safety-critical enforcement. Tool availability and execution checks must enforce mode and capability policy.
- Keep existing local-agent tests passing. Add focused tests for each sub-milestone before opening its implementation PR.

## Design Recommendations

### Todo State

Reuse Warp's existing todo primitives instead of creating a separate local-agent todo store:

- `AIAgentTodoList`
- `AIAgentTodo`
- `TodoOperation`
- `api::Message::UpdateTodos`
- `BlocklistAIHistoryEvent::UpdatedTodoList`
- Existing todo rendering in the AI block and plan/todo popup

M4.1 should treat the todo list as conversation-scoped runtime state. Existing conversation replay may reconstruct todos from history when the same conversation is restored, but M4.1 should not add cross-conversation persistence or a global todo file.

### Plan Mode

Reuse `UserQueryMode::Plan` and the existing `/plan` query path where practical, but add local-agent-specific state so the mode persists across provider rounds until `exit_plan_mode` returns a plan. Plan mode must hide and deny mutating tools at the system/executor layer. The provider should never be able to bypass the policy by calling a hidden tool name directly.

### Skills

Reuse the existing skill parser and `SkillManager` where practical, but constrain M4.3 local-agent behavior to Claude-style local filesystem paths:

- `~/.claude/skills/<name>/SKILL.md`
- `<project>/.claude/skills/<name>/SKILL.md`

Skill metadata can be injected into the local system prompt. Full `SKILL.md` content should be loaded only when the user invokes `/skill-name` or another explicit skill trigger. Skills are prompt-as-data, not executable plugins.

## External Reference Points

- Claude Code exposes a plan mode as a permission mode where the agent can inspect context and produce a plan before making changes.
- Claude Code skills are local folders with a `SKILL.md` file and frontmatter metadata; content is loaded when relevant instead of always being injected in full.
- The existing Warp codebase already has todo list UI, `UserQueryMode::Plan`, slash-command skill invocation, and skill discovery primitives.

## Non-Goals For M4

- Do not add remote skill marketplace sync, installation, update checks, or trust cache.
- Do not let skill packages execute shell commands, write files, or run scripts automatically.
- Do not add automatic plan approval, autonomous execution after plan exit, or background task execution.
- Do not classify arbitrary MCP tools as read-only unless M4.2 adds an explicit trusted read-only signal. Conservative default is to hide/deny MCP tools in plan mode.
- Do not add codebase indexing, subagents, web search, fetch, image output, or new telemetry channels.
