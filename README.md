# shiftpaper

[![CI](https://github.com/CPritch/shiftpaper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CPritch/shiftpaper/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CPritch/shiftpaper)](https://github.com/CPritch/shiftpaper/releases/latest)
[![AUR](https://img.shields.io/aur/version/shiftpaper)](https://aur.archlinux.org/packages/shiftpaper)
[![License: MIT](https://img.shields.io/github/license/CPritch/shiftpaper)](LICENSE)

Parallax wallpaper daemon for Wayland. Uses monocular depth estimation to generate a depth map from any image, then shifts the wallpaper layers based on cursor position. The effect is subtle but satisfying.


![Example Tiger Wallpaper](header.gif)

## How it works

Two binaries:

- `shiftpaper` (CLI) takes a source image, runs a monocular depth model (MoGe-2 by default) via ONNX Runtime, and writes a color + 16-bit depth PNG pair to a cache directory.
- `shiftpaperd` (daemon) loads the pre-baked pair and renders a parallax-displaced wallpaper on wlr-layer-shell surfaces using wgpu/Vulkan.

The daemon has no ML dependencies. All inference happens in the CLI.

## Dependencies

- Wayland compositor with wlr-layer-shell support (see [Compositors](#compositors))
- Vulkan-capable GPU
- ONNX Runtime (`onnxruntime` on Arch, or `onnxruntime-cuda` to bake on an NVIDIA GPU)

Building from source additionally requires a Rust toolchain.

## Install

Arch Linux (AUR):

```
paru -S shiftpaper        # or yay, pikaur, etc.
```

`shiftpaper-git` tracks the latest `main` instead.

From source:

```
cargo install --path crates/cli
cargo install --path crates/daemon
```

## Quick start

```
shiftpaper fetch-model                      # downloads the depth model (~420 MB)
shiftpaper set ~/Pictures/wallpaper.jpg
systemctl --user enable --now shiftpaperd
```

To change wallpaper, `shiftpaper set` another image. The new one morphs in through the depth of the scene.

For a slideshow, run `shiftpaper slideshow ~/Pictures/walls --interval 10m`. Add `--shuffle` to mix up the order, and `shiftpaper slideshow --stop` keeps whichever image is showing.

`shiftpaper fetch-model --list` shows the other depth models. Their licences are in [THIRD_PARTY.md](THIRD_PARTY.md).

From a source install, run `shiftpaperd` directly, or copy `shiftpaperd.service` into `~/.config/systemd/user/` and point its `ExecStart` at your binary.

## Tracking modes

- `pointer` (default): works on any wlr-layer-shell compositor. The wallpaper only moves while the cursor is over the desktop.
- `hyprland`: follows the cursor everywhere, even over windows. Hyprland only, and it lets the daemon see the cursor over other apps.

Switch with `shiftpaper mode hyprland`, then restart the daemon.

## Docs

Proper docs are still to come. Until then, `shiftpaper --help` covers the CLI, and the config lives at `~/.config/shiftpaper/config.toml`.

## Compositors

Only tested on Hyprland so far. It should work on other compositors with wlr-layer-shell, such as Sway, River, Wayfire, niri and labwc. GNOME and KDE Plasma don't support it.

## License

MIT
