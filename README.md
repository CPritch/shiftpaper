# shiftpaper

Parallax wallpaper daemon for Wayland. Uses monocular depth estimation to generate a depth map from any image, then shifts the wallpaper layers based on cursor position. The effect is subtle but satisfying.

Experimental. Built and tested on a single machine (Arch, Hyprland, RTX 5060). Expect rough edges.

![Example Tiger Wallpaper](header.gif)

## How it works

Two binaries:

- `shiftpaper` (CLI) takes a source image, runs Depth Anything V2/V3 inference via ONNX Runtime, and writes a color + 16-bit depth PNG pair to a cache directory.
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

From source:

```
cargo install --path crates/cli
cargo install --path crates/daemon
```

## Quick start

```
shiftpaper fetch-model                      # downloads the depth model (~99 MB)
shiftpaper set ~/Pictures/wallpaper.jpg
systemctl --user enable --now shiftpaperd
```

To change wallpaper, `shiftpaper set` another image and run `systemctl --user reload shiftpaperd`.

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
