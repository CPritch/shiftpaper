//! The config.toml schema shared by the `shiftpaper` CLI, which writes the
//! file, and the `shiftpaperd` daemon, which reads it.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::ErrorKind;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub daemon: DaemonConfig,
    /// None until the first `shiftpaper set`.
    pub wallpaper: Option<WallpaperConfig>,
    #[serde(default)]
    pub monitor: Vec<MonitorOverride>,
    /// None until the first `shiftpaper fetch-model` or `set`.
    pub inference: Option<InferenceConfig>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub cursor_poll_hz: NonZeroU32,
    pub parallax_intensity: f32,
    pub idle_timeout_secs: u64,
    pub battery_threshold: u8,
    pub tracking_mode: TrackingMode,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
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

impl TrackingMode {
    /// The value as written in config.toml.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pointer => "pointer",
            Self::Hyprland => "hyprland",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct WallpaperConfig {
    pub color: PathBuf,
    pub depth: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MonitorOverride {
    pub name: String,
    pub color: Option<PathBuf>,
    pub depth: Option<PathBuf>,
    pub parallax_intensity: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            cursor_poll_hz: const { NonZeroU32::new(60).unwrap() },
            parallax_intensity: 0.025,
            idle_timeout_secs: 300,
            battery_threshold: 20,
            tracking_mode: TrackingMode::default(),
        }
    }
}

impl Config {
    /// Load config.toml, or None if it doesn't exist. A leading `~` in
    /// any path is expanded to the home directory.
    pub fn load() -> Result<Option<Self>> {
        let path = path();
        let Some(text) = read_text(&path)? else {
            return Ok(None);
        };

        let mut cfg: Config =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;

        if let Some(w) = &mut cfg.wallpaper {
            w.color = expand_tilde(&w.color);
            if let Some(p) = &mut w.depth {
                *p = expand_tilde(p);
            }
        }
        for m in &mut cfg.monitor {
            if let Some(p) = &mut m.color {
                *p = expand_tilde(p);
            }
            if let Some(p) = &mut m.depth {
                *p = expand_tilde(p);
            }
        }
        if let Some(i) = &mut cfg.inference {
            i.model_path = expand_tilde(&i.model_path);
        }

        Ok(Some(cfg))
    }

    /// True if at least one output has a color image to show.
    pub fn has_wallpaper(&self) -> bool {
        self.wallpaper.is_some() || self.monitor.iter().any(|m| m.color.is_some())
    }

    fn override_for(&self, output_name: &str) -> Option<&MonitorOverride> {
        self.monitor.iter().find(|m| m.name == output_name)
    }

    /// The output's own color image, else the global wallpaper's.
    pub fn color_for(&self, output_name: &str) -> Option<&Path> {
        self.override_for(output_name)
            .and_then(|m| m.color.as_deref())
            .or_else(|| self.wallpaper.as_ref().map(|w| w.color.as_path()))
    }

    pub fn depth_for(&self, output_name: &str) -> Option<PathBuf> {
        let monitor = self.override_for(output_name);
        if let Some(d) = monitor.and_then(|m| m.depth.as_deref()) {
            return Some(d.to_path_buf());
        }
        // The global depth map belongs to the global color image, so a
        // monitor with its own color must infer its depth from that instead.
        if let Some(c) = monitor.and_then(|m| m.color.as_deref()) {
            return Some(infer_depth_path(c));
        }
        let wallpaper = self.wallpaper.as_ref()?;
        Some(
            wallpaper
                .depth
                .clone()
                .unwrap_or_else(|| infer_depth_path(&wallpaper.color)),
        )
    }

    pub fn intensity_for(&self, output_name: &str) -> f32 {
        self.override_for(output_name)
            .and_then(|m| m.parallax_intensity)
            .unwrap_or(self.daemon.parallax_intensity)
    }
}

/// `$XDG_CONFIG_HOME/shiftpaper/config.toml`, else
/// `~/.config/shiftpaper/config.toml`.
pub fn path() -> PathBuf {
    // The XDG spec says to ignore an empty or relative XDG_CONFIG_HOME.
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from)
        && xdg.is_absolute()
    {
        xdg.join("shiftpaper/config.toml")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/shiftpaper/config.toml")
    } else {
        PathBuf::from("config.toml")
    }
}

/// The file's contents, or None if it doesn't exist.
pub fn read_text(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("failed to read {}", path.display())),
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

fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(stripped) = p.strip_prefix("~")
        && let Some(home) = std::env::var_os("HOME")
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
    fn missing_daemon_keys_use_defaults() {
        let cfg = parse(
            r#"
            [daemon]
            tracking_mode = "hyprland"
            "#,
        );
        assert_eq!(cfg.daemon.tracking_mode, TrackingMode::Hyprland);
        assert_eq!(cfg.daemon.parallax_intensity, 0.025);
        assert_eq!(cfg.daemon.idle_timeout_secs, 300);
        assert_eq!(cfg.daemon.battery_threshold, 20);
    }

    #[test]
    fn tracking_mode_strings_match_the_parser() {
        for mode in [TrackingMode::Pointer, TrackingMode::Hyprland] {
            let cfg = parse(&format!("[daemon]\ntracking_mode = \"{}\"", mode.as_str()));
            assert_eq!(cfg.daemon.tracking_mode, mode);
        }
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = toml::from_str::<Config>(
            r#"
            [daemon]
            parallax_intensty = 0.05
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("parallax_intensty"));
    }

    #[test]
    fn unknown_sections_are_rejected() {
        let err = toml::from_str::<Config>(
            r#"
            [[monitors]]
            name = "DP-1"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("monitors"));
    }

    #[test]
    fn zero_poll_rate_is_rejected() {
        let result = toml::from_str::<Config>(
            r#"
            [daemon]
            cursor_poll_hz = 0
            "#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn config_before_first_set_has_no_wallpaper() {
        let cfg = parse(
            r#"
            [inference]
            model_path = "/m/model.onnx"
            "#,
        );
        assert_eq!(
            cfg.inference.as_ref().unwrap().model_path,
            PathBuf::from("/m/model.onnx")
        );
        assert!(!cfg.has_wallpaper());
    }

    #[test]
    fn monitor_only_config_has_wallpaper() {
        let cfg = parse(
            r#"
            [[monitor]]
            name = "DP-1"
            color = "/w/external.color.png"
            "#,
        );
        assert!(cfg.has_wallpaper());
        assert_eq!(
            cfg.color_for("DP-1"),
            Some(Path::new("/w/external.color.png"))
        );
        assert_eq!(cfg.color_for("eDP-1"), None);
        assert_eq!(cfg.depth_for("eDP-1"), None);
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
            Some(PathBuf::from("/w/global.depth16.png"))
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
        assert_eq!(
            cfg.color_for("DP-1"),
            Some(Path::new("/w/external.color.png"))
        );
        assert_eq!(
            cfg.depth_for("DP-1"),
            Some(PathBuf::from("/w/external.depth16.png"))
        );
        assert_eq!(
            cfg.color_for("eDP-1"),
            Some(Path::new("/w/global.color.png"))
        );
        assert_eq!(
            cfg.depth_for("eDP-1"),
            Some(PathBuf::from("/w/global.depth16.png"))
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
            Some(PathBuf::from("/w/hand_made.depth16.png"))
        );
    }

    #[test]
    fn intensity_falls_back_to_daemon_setting() {
        let cfg = parse(
            r#"
            [daemon]
            parallax_intensity = 0.05

            [[monitor]]
            name = "DP-1"
            parallax_intensity = 0.1
            "#,
        );
        assert_eq!(cfg.intensity_for("DP-1"), 0.1);
        assert_eq!(cfg.intensity_for("eDP-1"), 0.05);
    }
}
