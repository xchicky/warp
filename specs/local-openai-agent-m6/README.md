# Local OpenAI-Compatible Agent M6 Follow-Ups

## Summary

M6 covers post-wave-1 dogfood follow-ups that improve daily local-agent ergonomics without changing the wave 2 promotion plan. The first priority is safe command auto-execution for the Base Agent route. The second priority is making the local agent usable when a foreground TUI owns the terminal.

Dogfood validation for these follow-ups must use `WarpLocal.app` / `Channel::Local`, not `WarpOss.app`, so promoted dogfood flags are available during manual verification.

## Milestones

1. [M6.1: Auto-Execute Safe Commands](M6.1-auto-execute-safe-commands.md)
2. [M6.2: Local Agent in Full Terminal Use](M6.2-local-agent-in-cli-mode.md)
   - [M6.2 implementation spec](M6.2-implementation.md)
   - [M6.2 Option A research notes](_research/M6.2-option-a-research.md)

## Priority

M6.1 is first. It addresses the highest-friction local-agent workflow: multi-step read-only inspection currently stalls on every `run_shell_command` approval even when the command is harmless.

M6.2 follows after M6.1. It is larger because Full Terminal Use is the CLI-agent/TUI path, not the Base Agent OpenAI-compatible tool path.

## Gates

- Spec-only PR first, then implementation PRs after PM approval.
- One feature flag per implementation PR.
- Manual dogfood verification is required on `WarpLocal.app`.
- For bugs in either milestone, collect a runtime trace before committing a fix.
