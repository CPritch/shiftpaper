use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub daemon: DaemonConfig,
    pub wallpaper: WallpaperConfig,
    #[serde(default)]
    pub monitor: Vec<MonitorOverride>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DaemonConfig {
    #[serde(default = "default_poll_hz")]
    pub cursor_poll_hz: u32,
    #[serde(default = "default_intensity")]
    pub parallax_intensity: f32,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_secs: u64,
    #[serde(default = "default_battery_threshold")]
    pub battery_threshold: u8,
    #[serde(default)]
    pub tracking_mode: TrackingMode,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrackingMode {
    /// Wayland-native pointer events. Default. Works on any
    /// wlr-layer-shell compositor; renders only when the cursor is
    /// over visible desktop.
    #[default]
    Pointer,
    /// Hyprland IPC global cursor polling. Hyprland-only; sees the
    /// cursor even when windows cover the desktop.
    Hyprland,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WallpaperConfig {
    pub color: PathBuf,
    #[serde(default)]
    pub depth: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MonitorOverride {
    pub name: String,
    pub color: Option<PathBuf>,
    pub depth: Option<PathBuf>,
    pub parallax_intensity: Option<f32>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            cursor_poll_hz: default_poll_hz(),
            parallax_intensity: default_intensity(),
            idle_timeout_secs: default_idle_timeout(),
            battery_threshold: default_battery_threshold(),
            tracking_mode: TrackingMode::default(),
        }
    }
}

fn default_poll_hz() -> u32 {
    60
}
fn default_intensity() -> f32 {
    0.025
}
fn default_idle_timeout() -> u64 {
    300
}
fn default_battery_threshold() -> u8 {
    20
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_path();
        tracing::debug!(?path, "looking for config file");

        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config at {}", path.display()))?;

        let mut cfg: Config = toml::from_str(&text).with_context(|| "failed to parse config")?;

        cfg.wallpaper.color = expand_tilde(&cfg.wallpaper.color);
        if let Some(ref mut p) = cfg.wallpaper.depth {
            *p = expand_tilde(p);
        }
        for m in &mut cfg.monitor {
            if let Some(ref mut p) = m.color {
                *p = expand_tilde(p);
            }
            if let Some(ref mut p) = m.depth {
                *p = expand_tilde(p);
            }
        }

        Ok(cfg)
    }

    pub fn color_for(&self, output_name: &str) -> &Path {
        self.monitor
            .iter()
            .find(|m| m.name == output_name)
            .and_then(|m| m.color.as_deref())
            .unwrap_or(&self.wallpaper.color)
    }

    pub fn depth_for(&self, output_name: &str) -> PathBuf {
        let monitor = self.monitor.iter().find(|m| m.name == output_name);
        if let Some(d) = monitor.and_then(|m| m.depth.as_deref()) {
            return d.to_path_buf();
        }
        // The global depth map belongs to the global color image, so a
        // monitor with its own color must infer its depth from that instead.
        if let Some(c) = monitor.and_then(|m| m.color.as_deref()) {
            return infer_depth_path(c);
        }
        if let Some(d) = &self.wallpaper.depth {
            return d.clone();
        }
        infer_depth_path(&self.wallpaper.color)
    }

    pub fn intensity_for(&self, output_name: &str) -> f32 {
        self.monitor
            .iter()
            .find(|m| m.name == output_name)
            .and_then(|m| m.parallax_intensity)
            .unwrap_or(self.daemon.parallax_intensity)
    }
}

fn infer_depth_path(color: &Path) -> PathBuf {
    let s = color.to_string_lossy();
    if let Some(stripped) = s.strip_suffix(".color.png") {
        return PathBuf::from(format!("{stripped}.depth16.png"));
    }
    let mut p = color.to_path_buf();
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    p.set_file_name(format!("{stem}.depth16.png"));
    p
}

fn config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg).join("shiftpaper/config.toml")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config/shiftpaper/config.toml")
    } else {
        PathBuf::from("config.toml")
    }
}

fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(stripped) = p.strip_prefix("~")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(stripped);
    }
    p.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Config {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn infers_depth_from_baked_color_name() {
        assert_eq!(
            infer_depth_path(Path::new("/c/abc.color.png")),
            PathBuf::from("/c/abc.depth16.png")
        );
    }

    #[test]
    fn infers_depth_from_other_image_names() {
        assert_eq!(
            infer_depth_path(Path::new("/p/photo.jpg")),
            PathBuf::from("/p/photo.depth16.png")
        );
    }

    #[test]
    fn global_depth_inferred_when_unset() {
        let cfg = parse(
            r#"
            [wallpaper]
            color = "/w/global.color.png"
            "#,
        );
        assert_eq!(
            cfg.depth_for("DP-1"),
            PathBuf::from("/w/global.depth16.png")
        );
    }

    #[test]
    fn monitor_color_override_does_not_use_global_depth() {
        let cfg = parse(
            r#"
            [wallpaper]
            color = "/w/global.color.png"
            depth = "/w/global.depth16.png"

            [[monitor]]
            name = "DP-1"
            color = "/w/external.color.png"
            "#,
        );
        assert_eq!(cfg.color_for("DP-1"), Path::new("/w/external.color.png"));
        assert_eq!(
            cfg.depth_for("DP-1"),
            PathBuf::from("/w/external.depth16.png")
        );
        assert_eq!(cfg.color_for("eDP-1"), Path::new("/w/global.color.png"));
        assert_eq!(
            cfg.depth_for("eDP-1"),
            PathBuf::from("/w/global.depth16.png")
        );
    }

    #[test]
    fn monitor_depth_override_wins() {
        let cfg = parse(
            r#"
            [wallpaper]
            color = "/w/global.color.png"

            [[monitor]]
            name = "DP-1"
            color = "/w/external.color.png"
            depth = "/w/hand_made.depth16.png"
            "#,
        );
        assert_eq!(
            cfg.depth_for("DP-1"),
            PathBuf::from("/w/hand_made.depth16.png")
        );
    }

    #[test]
    fn intensity_falls_back_to_daemon_setting() {
        let cfg = parse(
            r#"
            [daemon]
            parallax_intensity = 0.05

            [wallpaper]
            color = "/w/global.color.png"

            [[monitor]]
            name = "DP-1"
            parallax_intensity = 0.1
            "#,
        );
        assert_eq!(cfg.intensity_for("DP-1"), 0.1);
        assert_eq!(cfg.intensity_for("eDP-1"), 0.05);
    }
}
