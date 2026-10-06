# Troubleshooting

## Seeing what the daemon is up to

```
journalctl --user -u shiftpaperd -f
```

For more detail, stop the service and run the daemon yourself:

```
systemctl --user stop shiftpaperd
RUST_LOG=shiftpaperd=debug shiftpaperd
```

## The daemon won't start

- If the log says to run `shiftpaper set <image>` first, it has nothing to show yet. Set a wallpaper, then start it again.
- systemd stops trying after five failed starts in a minute. Once you've fixed the cause, run `systemctl --user reset-failed shiftpaperd` and start it again.
- If you installed with cargo, systemd can't find `shiftpaperd` unless the service's `ExecStart` has its full path. See [Install](../README.md#install).
- shiftpaper needs a compositor with wlr-layer-shell. Hyprland is the only one tested so far, but Sway, River, Wayfire, niri and labwc should work. GNOME and KDE Plasma don't support it.

## The wallpaper doesn't move

- In the default `pointer` mode, it only moves while the cursor is over the desktop. On Hyprland, `shiftpaper mode hyprland` makes it follow the cursor over windows too (restart the daemon afterwards).
- It pauses after a while without input, and on low battery. See `idle_timeout_secs` and `battery_threshold` in [configuration](configuration.md).
- Some images don't have much depth to them, like a flat field or a far-off view. Images with something close up, like a tree or a rock, show it off best. You can also turn `parallax_intensity` up a little.

## Baking fails

- "no model found" means there's no depth model yet. Run `shiftpaper fetch-model`.
- "failed to load ONNX Runtime" means `onnxruntime` (or `onnxruntime-cuda`) is missing or broken. Install it, and make sure the rest of your system is up to date, as a half-finished upgrade can break it. `ORT_DYLIB_PATH` can point at a particular `libonnxruntime.so`.
