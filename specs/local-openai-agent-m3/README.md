# Local OpenAI-Compatible Agent: M3 Image Input and Token-Cost Telemetry

## Summary

M3 adds two independent capabilities to the local OpenAI-compatible agent:

1. Image input passthrough for vision-capable OpenAI-compatible providers.
2. Local token and cost telemetry from provider-reported usage.

The features touch different data directions and should be split into two sub-milestones. Image input sends additional user-provided bytes to the provider. Token-cost telemetry reads provider usage metadata back from responses and surfaces a local estimate to the user. Keep them separate so each can be reviewed and reverted independently.

## Recommended Split

1. [M3.1-image-input-passthrough.md](M3.1-image-input-passthrough.md): feature flag, existing image attachment reuse, safe image normalization, OpenAI-compatible multimodal message encoding, and capability fallback.
2. [M3.2-token-cost-telemetry.md](M3.2-token-cost-telemetry.md): feature flag, streaming usage extraction, local pricing registry, `StreamFinished` usage metadata, and usage footer display.

M3.1 and M3.2 are technically independent after M2.4. They can be developed sequentially for lower review load or in parallel if agent1 has a clean branch per sub-milestone. If developed sequentially, prefer M3.1 first because it exercises the request message shape while M3.2 exercises the response finalization shape.

## Shared Constraints

- Both features must be guarded by default-off feature flags:
  - `FeatureFlag::LocalAgentImageInput`
  - `FeatureFlag::LocalAgentCostTelemetry`
- Do not change remote/Oz provider behavior except where shared UI or persistence code needs a narrow extension.
- Do not introduce a privacy or product telemetry upload path for local usage or image bytes.
- Keep local provider secrets, image bytes, base64 data URLs, and raw usage responses out of logs.
- Existing M2 local-agent behavior and tests must continue to pass.

## Research Notes

Current repo state:

- User images already enter request context as `AIAgentContext::Image(ImageContext)` through pending image attachments in `BlocklistAIContextModel`.
- `ImageContext` already stores base64 data, MIME type, filename, and has a redacted `Debug` implementation.
- Local direct currently drops `AIAgentContext::Image(_)` in `local_context_section` and serializes OpenAI chat messages with `content: String`.
- Shared image helpers already exist in `app/src/util/image.rs`: supported MIME whitelist, image count/size constants, MIME sniffing, and resize processing.
- Local direct currently emits `StreamFinished` with empty token usage and zero credits.
- Existing conversation usage UI reads `conversation_usage_metadata`, `request_cost`, and `token_usage` from `StreamFinished` and renders the usage footer when any usage exists.

External reference points:

- OpenAI-compatible chat providers commonly accept multimodal user message content as an array containing text parts and `image_url` parts, where `image_url.url` may be a `data:<mime>;base64,<data>` URL.
- OpenAI-compatible streaming usage is commonly exposed via `usage` on a final stream chunk, often enabled by `stream_options: { "include_usage": true }`.
- Provider pricing changes frequently. M3 should ship a small local, overrideable pricing registry and treat missing prices as unknown rather than blocking usage display.

## Non-Goals For M3

- Do not add file-id upload APIs for local providers in M3. Data URL passthrough is the first implementation target.
- Do not add OCR, image captioning, image generation, image output, or MCP resource image handling.
- Do not add TodoWrite, plan mode, skill loading, codebase index, subagents, web search, or fetch.
- Do not send local token/cost estimates to Warp servers or third-party analytics.
