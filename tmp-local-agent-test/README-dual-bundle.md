# Dual Local macOS Bundles: Warp and Warp Dev

## Purpose

Use `Warp.app` as the stable daily dogfood bundle and `Warp Dev.app` for testing new local-agent changes in isolation. Both are local-channel builds, but they use different bundle identifiers, URL schemes, and data profiles so Dev testing does not overwrite daily-use state.

## Stable Bundle

- App path: `/Users/yizhang/Applications/Warp.app`
- Display name: `Warp`
- Bundle id: `dev.warp.Warp-Local`
- URL scheme: `warp://`
- Data dir/profile: `~/.warp-local`
- Build command:

```bash
PATH="/tmp/warp-channel-stub-bin:$PATH" WARP_BIN_NAME=warp WARP_CHANNEL=local FEATURES=gui,release_bundle ./script/macos/run --dont-open
```

Install Stable:

```bash
rm -rf /Users/yizhang/Applications/Warp.app
ditto target/debug/bundle/osx/Warp.app /Users/yizhang/Applications/Warp.app
```

## Dev Bundle

- App path: `/Users/yizhang/Applications/Warp Dev.app`
- Display name: `Warp Dev`
- Bundle id: `dev.warp.Warp-Local-Dev`
- URL scheme: `warpdev://`
- Intended isolated data profile: `~/.warp-local-dev` when launched with `WARP_DATA_PROFILE=local-dev`
- Build command:

```bash
PATH="/tmp/warp-channel-stub-bin:$PATH" WARP_BIN_NAME=warp WARP_CHANNEL=local-dev FEATURES=gui,release_bundle ./script/macos/run --dont-open
```

Install Dev with backup rotation:

```bash
rm -rf /Users/yizhang/Applications/Warp\ Dev.app.previous-3
mv /Users/yizhang/Applications/Warp\ Dev.app.previous-2 /Users/yizhang/Applications/Warp\ Dev.app.previous-3 2>/dev/null || true
mv /Users/yizhang/Applications/Warp\ Dev.app.previous-1 /Users/yizhang/Applications/Warp\ Dev.app.previous-2 2>/dev/null || true
mv /Users/yizhang/Applications/Warp\ Dev.app /Users/yizhang/Applications/Warp\ Dev.app.previous-1 2>/dev/null || true
ditto target/debug/bundle/osx/Warp\ Dev.app /Users/yizhang/Applications/Warp\ Dev.app
```

List Dev backups:

```bash
ls -ld /Users/yizhang/Applications/Warp\ Dev.app.previous-* 2>/dev/null || true
```

## Data Profile Launching

`script/macos/run` sets `WARP_DATA_PROFILE=local-dev` when it directly launches the `local-dev` bundle. V0 does not add a bundle-internal launcher wrapper, because replacing the bundled executable with a wrapper would be fragile with `cargo bundle`, rpath updates, and codesigning.

When launching the installed Dev app outside `script/macos/run`, use:

```bash
WARP_DATA_PROFILE=local-dev /Users/yizhang/Applications/Warp\ Dev.app/Contents/MacOS/warp
```

Double-clicking `Warp Dev.app` may not use the isolated `~/.warp-local-dev` profile in V0.

## Build Profile Note

Use dogfood/debug-assertion builds for both local bundles. `--release` is not valid for Dev data-profile isolation because `WARP_DATA_PROFILE` is intentionally ignored outside debug assertions.

## Promote and Rollback

Promote flow: once the user reports Dev is stable, PM asks agent1 to rebuild the same SHA as Stable and install `/Users/yizhang/Applications/Warp.app`.

Rollback flow: keep the last three `/Users/yizhang/Applications/Warp Dev.app.previous-N` backups and restore one with `ditto` if needed.

## Inspect Info.plist

```bash
/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' /Users/yizhang/Applications/Warp.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleDisplayName' /Users/yizhang/Applications/Warp.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleName' /Users/yizhang/Applications/Warp.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleURLTypes' /Users/yizhang/Applications/Warp.app/Contents/Info.plist

/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' /Users/yizhang/Applications/Warp\ Dev.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleDisplayName' /Users/yizhang/Applications/Warp\ Dev.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleName' /Users/yizhang/Applications/Warp\ Dev.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c 'Print :CFBundleURLTypes' /Users/yizhang/Applications/Warp\ Dev.app/Contents/Info.plist
```
