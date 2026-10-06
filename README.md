<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/icon/readme-dark.svg">
  <img src="assets/icon/readme-light.svg" alt="shiftpaper icon" width="96">
</picture>

# shiftpaper

[![CI](https://github.com/CPritch/shiftpaper/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CPritch/shiftpaper/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CPritch/shiftpaper)](https://github.com/CPritch/shiftpaper/releases/latest)
[![AUR](https://img.shields.io/aur/version/shiftpaper)](https://aur.archlinux.org/packages/shiftpaper)
[![License: MIT](https://img.shields.io/github/license/CPritch/shiftpaper)](LICENSE)

Parallax wallpaper daemon for Wayland. Uses monocular depth estimation to generate a depth map from any image, then shifts the wallpaper layers based on cursor position. The effect is subtle but satisfying.


![A forest path, a beach of round stones and a rock face shifting with the cursor, and the transitions between them](assets/reel.avif)

<sub>Photos by [Charles Black](https://commons.wikimedia.org/wiki/File:Rotting_leaves_on_a_forest_path_(Unsplash).jpg), [Zeny Rosalina](https://commons.wikimedia.org/wiki/File:Smooth_Round_Rocks_Ocean_(Unsplash).jpg) and [Lionello DelPiccolo](https://commons.wikimedia.org/wiki/File:Climbing_in_Golden_Gate_Canyon_(Unsplash).jpg), CC0 via Wikimedia Commons.</sub>

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

To run it with systemd, copy `shiftpaperd.service` into `~/.config/systemd/user/` and change its `ExecStart` to `%h/.cargo/bin/shiftpaperd`.

## Quick start

Every command has its own help, so `shiftpaper help` is a good place to look around.

```
shiftpaper fetch-model                      # download the depth model (~420 MB)
shiftpaper set ~/Pictures/wallpaper.jpg     # make an image your wallpaper
systemctl --user enable --now shiftpaperd   # start the daemon
```

Then try:

```
shiftpaper set ~/Pictures/another.jpg       # change it
shiftpaper slideshow ~/Pictures/walls       # show a folder of images in turn
shiftpaper transition portal                # try a different transition
```

## Docs

The [docs](docs/) cover the config file, the transitions, depth models and troubleshooting.

## Compositors

Only tested on Hyprland so far. It should work on other compositors with wlr-layer-shell, such as Sway, River, Wayfire, niri and labwc. GNOME and KDE Plasma don't support it.

## License

MIT
