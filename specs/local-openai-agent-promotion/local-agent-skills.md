# LocalAgentSkills Promotion Plan

## Summary

Promote local Claude-style skill discovery and slash-triggered skill loading for the local OpenAI-compatible agent. Skills are prompt-as-data only and must not execute code automatically.

Recommended wave: 1.

## Current Status

- Implemented in M4.3 behind `FeatureFlag::LocalAgentSkills`, default off.
- Discovers user-level `~/.claude/skills/<name>/SKILL.md` and project-level `.claude/skills/<name>/SKILL.md`.
- Validates kebab-case names and description length.
- Project skills override user skills.
- Injects bounded metadata by default and loads full skill content only on explicit `/skill-name`.
- Enforces path containment for skill resources.

## Known Limitations

- Skill content can affect prompting quality but is not executable.
- Invalid skills are skipped with warnings rather than blocking the agent.
- Built-in slash command conflicts keep built-in command priority.

## Dogfood Gate

1. Confirm local-agent and `skills` module tests pass on current `master`.
2. Manually verify user-level and project-level skill discovery.
3. Verify `/skill-name rest of query` preserves the remaining user query text.
4. Verify path traversal and symlink escape fixtures fail closed.

## Preview Gate

1. Dogfood for one cadence cycle with no unsafe resource-loading issue.
2. Add user-facing docs for skill directory layout, frontmatter, and local-only behavior.
3. Verify warnings for invalid/shadowed skills are understandable.

## Stable Gate

1. Preview for one cadence cycle with no prompt-injection or path-containment issue.
2. Confirm skill marketplace or remote skill fetching remains out of scope.

## Rollback

Demote `LocalAgentSkills`. Existing files remain on disk but are not discovered or injected by the local agent.

## Recommended PR Order

Promote after image/todo basics and before plan mode, because skills improve planning quality but do not add tool authority.
