# LocalAgentImageInput Promotion Plan

## Summary

Promote image input passthrough for local OpenAI-compatible vision models. This feature expands provider input types but does not add local mutation, shell execution, MCP, or web access.

Recommended wave: 1.

## Current Status

- Implemented in M3.1 behind `FeatureFlag::LocalAgentImageInput`, default off.
- Reuses existing user attachment/context paths.
- Encodes supported images as OpenAI-compatible data URLs.
- Strips EXIF metadata before provider submission.
- Does not log data URLs or image bytes.
- Does not retry without images after a provider rejects multimodal input.

## Known Limitations

- Provider/model vision capability detection is conservative and may need updates for new OpenAI-compatible providers.
- Large images are bounded and may be rejected before provider request.
- Providers differ in support for `image_url` data URLs.

## Dogfood Gate

1. Confirm local-agent tests and image utility tests pass on current `master`.
2. Manually verify at least two local/compatible vision providers if available.
3. Verify a known non-vision model fails before provider request with a clear user-facing error.
4. Confirm EXIF/location metadata is stripped in test fixtures.

## Preview Gate

1. Dogfood for one cadence cycle with no image privacy/logging issue.
2. Document supported image types and provider caveats.
3. Verify repeated text with attached images is not deduped away from the current query.

## Stable Gate

1. Preview for one cadence cycle with no provider-compatibility severity-1/2 issue.
2. Confirm support docs explain how to remove images and resend for text-only answers.

## Rollback

Demote `LocalAgentImageInput`. Local providers return to text-only request payloads; no persisted data migration is required.

## Recommended PR Order

Promote in wave 1 after telemetry or todo write. It is user-visible, so dogfood should include manual provider testing rather than relying only on unit tests.
