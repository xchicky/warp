# LocalAgentWeb Promotion Plan

## Summary

Promote public web search and single-page fetch for the local OpenAI-compatible agent. This flag adds outbound HTTP through explicit `web_search` and `web_fetch` tools.

Recommended wave: 2.

## Current Status

- Implemented in M5.1 behind `FeatureFlag::LocalAgentWeb`, default off.
- `web_search` uses user-configured `WARP_LOCAL_AGENT_SEARXNG_URL`; no default Warp or third-party search API.
- `web_fetch` is GET-only with no browser, JavaScript, cookies, forms, login, or crawling.
- Enforces SSRF checks for loopback, RFC1918, link-local, IPv6 ULA, IPv4-mapped IPv6, unspecified/bind-style, metadata, and `.local` hosts.
- Rechecks DNS and redirect targets.
- Applies timeout, redirect limit, byte/char truncation, and content-type allowlist.
- Reuses existing web search/fetch status messages.
- Plan mode allows web tools as read-only public inspection tools.

## Known Limitations

- Requires a user-provided SearXNG endpoint for search.
- `web_fetch` does not execute JavaScript or fetch authenticated pages.
- Robots.txt handling is out of scope for V0 single-URL fetch.

## Dogfood Gate

1. Confirm local-agent and web module tests pass on current `master`.
2. Manually verify search against a local or internal SearXNG endpoint.
3. Manually verify SSRF deny fixtures including redirect-to-private and IPv4-mapped IPv6.
4. Confirm fetched URLs/content are not sent to product telemetry or unsafe logs.

## Preview Gate

1. Dogfood for one cadence cycle with no SSRF/privacy issue.
2. Decide whether preview requires an endorsed SearXNG setup guide.
3. Confirm plan-mode `web_fetch` does not need an extra confirmation gate after dogfood signal.

## Stable Gate

1. Preview for one cadence cycle with no severity-1/2 outbound network issue.
2. Confirm support docs explain endpoint configuration and unsupported pages.

## Rollback

Demote `LocalAgentWeb`. The environment variable may remain set, but the local agent stops advertising/executing `web_search` and `web_fetch`.

## Recommended PR Order

Promote after local read-only/context flags and before subagent preview. Web expands outbound surface area, so it should not be bundled with any other promotion.
