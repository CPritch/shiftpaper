# Changelog

## Unreleased

### Added

- Monitors plugged in while the daemon is running get a wallpaper.
- Fractionally scaled displays render at full resolution instead of being upscaled.
- Wallpapers can be set per monitor without a global one.
- `shiftpaper-git` on the AUR.

### Changed

- Wallpapers are cropped to fill the screen instead of stretched.
- Depth maps keep the image's aspect ratio, which brings out more detail. Existing wallpapers are re-baked on the next `shiftpaper set`.
- Rendering pauses at the battery threshold rather than below it.
- Unknown or invalid config keys are an error instead of being ignored.
- Clearer errors when the config or wallpaper is missing.
- The systemd unit gives up after a few failed starts instead of retrying forever.
- Much smaller source download for the AUR package.

### Fixed

- Lag on fast mouse movement, where the wallpaper kept moving after the cursor stopped. It was worst on a second monitor.
- Hyprland tracking on multi-monitor setups.
- Hyprland mode redrawing constantly even with the cursor still.
- Pointer mode freezing part-way when the cursor left the desktop.
- Depth and color slightly out of line near the screen edges.
- A per-monitor `color` using the global wallpaper's depth map.
- `shiftpaper set` hanging with no output when ONNX Runtime couldn't load. It now says why.
- Reloading after `shiftpaper mode` stopping the parallax until a restart.
- The daemon and CLI never showing their own log messages.

## 0.1.0 - 2026-05-15

Initial release.
