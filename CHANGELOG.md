# Changelog

## Unreleased

### Changed

- `portal` grows like a sphere from the surface under the cursor, so it spreads over that first and wraps round things nearer and further away.
- `dissolve` grows spheres from random places in the scene, like `portal` does from the cursor, further things tending to go first.
- `tide-in` and `tide-out` find the ground in the scene, so the water rises level with it in views from above or at an angle, instead of always from the bottom of the picture. Close-ups and scenes without clear ground work as before.

## 0.4.2 - 2026-10-06

### Fixed

- 0.4.1 reported its version as 0.4.0.

## 0.4.1 - 2026-10-06

### Added

- Man pages: `man shiftpaper`, one for each command (like `man shiftpaper-slideshow`), and `man shiftpaperd`. Tab completion for bash, zsh and fish too. The AUR packages install both.
- Docs, in `docs/`: the config file, the transitions with a clip of each, depth models, and troubleshooting.
- `shiftpaperd --help` and `--version`.

### Changed

- Every command's help ends with a few examples, and `shiftpaper help` lists the commands in the order you'd use them.
- A new README header, and a shorter quick start.

### Fixed

- `shiftpaperd` ignored its arguments, so even `shiftpaperd --help` started the daemon.

## 0.4.0 - 2026-10-05

### Added

- Slideshows. `shiftpaper slideshow` bakes a set of images, or folders of them, and the daemon moves through them on a timer, in order or with `--shuffle`. They pause while idle or on low battery, and `shiftpaper slideshow --stop` keeps whichever image is showing.
- Transitions. Changing wallpaper, whether by slideshow or `shiftpaper set`, morphs from one to the next through the depth of the scene, and the parallax carries on throughout. `shiftpaper transition <name> [--secs N]` picks which and how long it takes (3 seconds by default), or set `transition` and `transition_secs` under `[daemon]`. `shiftpaper transition --help` describes them all:
  - `sweep-in` (the default): a wave sweeps into the scene, near things first.
  - `sweep-out`: the background goes first and the nearest things last.
  - `morph`: everything at once, the shape of the scene a little ahead of its colours.
  - `dissolve`: patches appear at random, nearer things tending to go first.
  - `portal`: grows out from your cursor like a bubble in the scene, and follows it.
  - `tide-in`: rises like water, lowest places first.
  - `tide-out`: drains away like water, uncovering the highest places first.
- An icon, shown in the README and installed by the AUR packages.

### Changed

- `set` and `slideshow` apply straight away: they tell a running shiftpaperd to reload, so there's no `systemctl --user reload` to run.
- Changing wallpaper no longer freezes the parallax while a big image loads: it loads in the background, and only once however many monitors show it. Settings changes that don't change the wallpaper don't reload it.
- Wallpapers much bigger than the screen are scaled down as they load. They draw several times faster and use far less memory.
- Config values that parse but can't work, like a negative `transition_secs`, are reported when the config loads.

### Fixed

- Idle detection never started, so the daemon kept rendering while you were away.
- The `shiftpaper-git` AUR page still said 0.2.0. Each release now keeps it current.

## 0.3.0 - 2026-10-04

### Added

- MoGe-2 depth models, with MoGe-2 ViT-B as the new default. Existing installs keep their current model until you run `shiftpaper fetch-model`.
- `shiftpaper fetch-model --list` shows the downloadable models, and `fetch-model <name>` gets a specific one. The licence is shown before downloading.
- Depth Anything V3 Small and Base can be downloaded.
- THIRD_PARTY.md lists each model's licence.

### Changed

- `fetch-model --url` is gone. Use `--model` with any ONNX file instead.
- Wallpapers are re-baked once, because the cache now keys bakes by model.

### Fixed

- Depth Anything V3 output was flipped instead of converted to disparity, which flattened the foreground and pulled the sky forward.
- Baking the same image with a different model reused the first model's depth map.
- The AUR package failed to build on systems without libxkbcommon.

## 0.2.0 - 2026-10-04

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
