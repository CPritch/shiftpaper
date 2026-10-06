# Configuration

Settings live in `~/.config/shiftpaper/config.toml`. The `shiftpaper` commands write it for you, so you only need to open it for things they don't cover, like a different wallpaper on each monitor.

When a command changes something, the daemon picks it up straight away. If you edit the file yourself, tell the daemon to reload it:

```
systemctl --user reload shiftpaperd
```

A few settings need a restart instead (`systemctl --user restart shiftpaperd`). They're marked below.

Paths can start with `~`.

## [daemon]

```toml
[daemon]
parallax_intensity = 0.03
transition = "portal"
transition_secs = 2
```

| Setting | Default | What it does |
|---|---|---|
| `parallax_intensity` | `0.025` | How far the wallpaper moves. Past about `0.05`, the edges of the image start to show. |
| `transition` | `"sweep-in"` | How one wallpaper changes into the next. See [transitions](transitions.md). |
| `transition_secs` | `3` | How long a transition takes, in seconds. `0` switches straight away. |
| `tracking_mode` | `"pointer"` | How the cursor is followed. See below. Needs a restart. |
| `cursor_poll_hz` | `60` | How often `hyprland` mode checks where the cursor is. Needs a restart. |
| `idle_timeout_secs` | `300` | Pause after this many seconds without any input. Needs a restart. |
| `battery_threshold` | `20` | Pause on battery at or below this percentage. `0` never pauses. |

While paused, the wallpaper stays still and a slideshow waits.

### Tracking modes

- `pointer`, the default, works on any compositor shiftpaper supports. The wallpaper only moves while the cursor is over the desktop, which also saves battery.
- `hyprland` follows the cursor everywhere, even over windows. It only works on Hyprland, and it lets the daemon see where your cursor is over other apps.

To switch:

```
shiftpaper mode hyprland
systemctl --user restart shiftpaperd
```

## [wallpaper]

`shiftpaper set` fills this in.

```toml
[wallpaper]
color = "~/.cache/shiftpaper/wallpapers/<hash>.color.png"
```

| Setting | What it does |
|---|---|
| `color` | The baked image to show. |
| `depth` | Its depth map. If it's left out, the `.depth16.png` next to `color` is used. |

## [slideshow]

`shiftpaper slideshow` fills this in, and `shiftpaper set` takes it out again. While it's there, it's shown instead of `[wallpaper]`.

```toml
[slideshow]
images = [
    "~/.cache/shiftpaper/wallpapers/<hash>.color.png",
    "~/.cache/shiftpaper/wallpapers/<hash>.color.png",
]
interval_secs = 600
shuffle = false
```

| Setting | Default | What it does |
|---|---|---|
| `images` | | The baked images to show in turn. |
| `interval_secs` | `600` | How long each one shows, in seconds. |
| `shuffle` | `false` | Show them in a random order, different each time round. |

## [[monitor]]

Each `[[monitor]]` block changes one monitor. This one gets its own image and a stronger effect:

```toml
[[monitor]]
name = "DP-1"
color = "~/.cache/shiftpaper/wallpapers/<hash>.color.png"
parallax_intensity = 0.04
```

| Setting | What it does |
|---|---|
| `name` | The monitor's name, like `DP-1` or `eDP-1`. `hyprctl monitors` or `wlr-randr` will list them. |
| `color` | Its own baked image. `shiftpaper bake image.jpg` bakes one without setting it, and prints the path to use first. |
| `depth` | Its own depth map, if it isn't next to `color`. |
| `parallax_intensity` | Its own intensity. |

A monitor with its own `color` keeps it while a slideshow plays on the others.

## [inference]

`shiftpaper fetch-model` fills this in.

```toml
[inference]
model_path = "~/.local/share/shiftpaper/models/moge-2-vitb/model.onnx"
```

| Setting | What it does |
|---|---|
| `model_path` | The depth model used for baking. See [depth models](models.md). |
