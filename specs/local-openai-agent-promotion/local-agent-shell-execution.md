# LocalAgentShellExecution Promotion Plan

## Summary

Promote shell command execution requests for the local OpenAI-compatible agent. This is the highest-risk single tool flag because accepted actions can execute arbitrary commands in the user's environment.

Recommended wave: 3, last.

## Current Status

- Implemented in M1.3 behind `FeatureFlag::LocalAgentShellExecution`, default off.
- Advertises `run_shell_command` only when enabled.
- Requests visible out-of-band user approval; commands are not auto-executed by the provider loop.
- Reuses existing shell action/result conversion for local resume.
- Validates command and cwd before confirmation.
- Participates in the sequential/mutating tool pool.
- Plan mode denies shell execution even if the flag is enabled.
- Subagents do not receive shell execution in V0 child catalogs.

## Known Limitations

- Shell risk depends on command content and user approval quality.
- Provider output can suggest risky commands; the approval UI is the hard gate.
- Long-running command lifecycle remains governed by existing shell/action infrastructure.

## Dogfood Gate

1. Confirm local-agent shell tests pass on current `master`.
2. Manually verify approval cards for safe, risky, and invalid commands.
3. Verify rejection, cancellation, and action-result resume stay on the local provider route.
4. Verify plan mode and subagent child catalogs deny shell execution.

## Preview Gate

1. Dogfood for one cadence cycle with no approval bypass or accidental execution issue.
2. Confirm user-facing copy makes clear commands are pending user approval and not yet executed.
3. Confirm command/cwd logging is redacted or safe under existing policies.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 shell incident.
2. Confirm support and security review agree that existing approval UX is sufficient for default availability.

## Rollback

Demote `LocalAgentShellExecution`. Already-approved commands are outside the flag once execution has started, but new local-agent shell command requests must stop immediately.

## Recommended PR Order

Promote last. Do not promote in the same cadence cycle as subagent preview unless PM explicitly accepts the compounded risk.
