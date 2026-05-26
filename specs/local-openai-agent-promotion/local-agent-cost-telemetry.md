# LocalAgentCostTelemetry Promotion Plan

## Summary

Promote local provider token and estimated cost reporting for the local OpenAI-compatible agent. This is a low-risk observability feature because it does not add tool execution, file IO, or network IO beyond the provider request already made by the agent.

Recommended wave: 1.

## Current Status

- Implemented in M3.2 behind `FeatureFlag::LocalAgentCostTelemetry`, default off.
- Requests `stream_options.include_usage` only when enabled.
- Retries unsupported stream-options responses only before streamed output to avoid double-charging.
- Maps provider usage into BYOK usage metadata under the primary agent category.
- Estimates local provider cost from a local pricing registry and local overrides.
- Does not write to Warp billing, product telemetry, or `AIRequestUsageModel`.

## Known Limitations

- Cost is estimated, not billed or guaranteed.
- Some OpenAI-compatible providers omit usage in streaming responses.
- Pricing can become stale until the local registry or override is updated.

## Dogfood Gate

1. Confirm local-agent tests pass on current `master`.
2. Manually verify one provider that returns usage and one provider/path that omits usage.
3. Verify UI copy uses "estimated" and does not mention Warp credits.
4. Verify no local pricing override is cloud-synced.

## Preview Gate

1. Dogfood for one cadence cycle with no billing or privacy confusion reports.
2. Add release-note copy explaining that estimates are provider-cost estimates for BYOK/local providers.
3. Confirm unsupported usage paths degrade silently without blocking the conversation.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 issue.
2. Pricing registry effective dates are visible in code/spec review.
3. Support has a clear answer for provider mismatch or missing usage reports.

## Rollback

Demote `LocalAgentCostTelemetry` to the previous rollout state. Conversations should continue without usage/cost footer metadata. No data migration is required.

## Recommended PR Order

Promote this first in wave 1 because it improves dogfood observability for later flags without expanding local-agent authority.
