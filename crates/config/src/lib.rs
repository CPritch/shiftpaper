//! The config.toml schema shared by the `shiftpaper` CLI, which writes the
//! file, and the `shiftpaperd` daemon, which reads it.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::ErrorKind;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "man")]
pub mod man;
mod shuffle;

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub daemon: DaemonConfig,
    /// None until the first `shiftpaper set`.
    pub wallpaper: Option<WallpaperConfig>,
    /// Takes the place of [wallpaper] on every monitor without a color of
    /// its own.
    pub slideshow: Option<SlideshowConfig>,
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
    pub transition: Transition,
    /// How long a change of wallpaper takes. 0 switches straight away.
    pub transition_secs: f32,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum TrackingMode {
    /// The default. Works on any compositor, and moves only while the
    /// cursor is over the desktop.
    #[default]
    Pointer,
    /// Follows the cursor over windows too. Hyprland only.
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

/// How one wallpaper changes into the next.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "kebab-case")]
pub enum Transition {
    /// A wave sweeps into the scene: the new wallpaper's nearest things
    /// appear first, in front of the old one, and it fills in towards the
    /// distance.
    #[default]
    SweepIn,
    /// A wave sweeps out of the scene: the old wallpaper's background gives
    /// way to the new one first, and its nearest things go last.
    SweepOut,
    /// Everything changes together, the shape of the scene a little ahead
    /// of its colours.
    Morph,
    /// Patches of the new wallpaper appear at random, nearer things
    /// tending to go first.
    Dissolve,
    /// The new wallpaper grows out from the cursor like a sphere in the
    /// scene, and follows the cursor if it moves.
    Portal,
    /// The new wallpaper rises through the old like a tide coming in,
    /// lowest places first.
    TideIn,
    /// The old wallpaper drains away like a tide going out, uncovering the
    /// new one from the highest places down.
    TideOut,
}

impl Transition {
    pub const ALL: [Self; 7] = [
        Self::SweepIn,
        Self::SweepOut,
        Self::Morph,
        Self::Dissolve,
        Self::Portal,
        Self::TideIn,
        Self::TideOut,
    ];

    /// The value as written in config.toml.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SweepIn => "sweep-in",
            Self::SweepOut => "sweep-out",
            Self::Morph => "morph",
            Self::Dissolve => "dissolve",
            Self::Portal => "portal",
            Self::TideIn => "tide-in",
            Self::TideOut => "tide-out",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct WallpaperConfig {
    pub color: PathBuf,
    pub depth: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SlideshowConfig {
    /// Baked color images, shown in turn. Each one's depth map is found
    /// the same way as for a [wallpaper] without a depth.
    pub images: Vec<PathBuf>,
    #[serde(default = "default_interval")]
    pub interval_secs: NonZeroU32,
    /// Show the images in a random order, different each time round.
    #[serde(default)]
    pub shuffle: bool,
}

fn default_interval() -> NonZeroU32 {
    const { NonZeroU32::new(600).unwrap() }
}

impl SlideshowConfig {
    /// Which slide to show at `now`. It's worked out from the clock rather
    /// than counted, so the daemon carries on in step after a restart or a
    /// sleep, and the CLI can tell which slide is showing.
    pub fn due(&self, now: SystemTime) -> usize {
        let len = self.images.len() as u64;
        let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let step = secs / u64::from(self.interval_secs.get());
        let position = (step % len) as usize;
        if self.shuffle {
            shuffle::order(step / len, self.images.len())[position]
        } else {
            position
        }
    }

    /// The color and depth images of the `index`th slide.
    pub fn slide(&self, index: usize) -> (&Path, PathBuf) {
        let color = &self.images[index];
        (color, infer_depth_path(color))
    }
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
            transition: Transition::default(),
            transition_secs: 3.0,
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
        cfg.validate()
            .with_context(|| format!("invalid {}", path.display()))?;

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
        if let Some(s) = &mut cfg.slideshow {
            for p in &mut s.images {
                *p = expand_tilde(p);
            }
        }

        Ok(Some(cfg))
    }

    /// Catch values that parse but make no sense.
    fn validate(&self) -> Result<()> {
        let secs = self.daemon.transition_secs;
        anyhow::ensure!(
            Duration::try_from_secs_f32(secs).is_ok(),
            "transition_secs must be a number of seconds, 0 or more, not {secs}"
        );
        if let Some(s) = &self.slideshow {
            anyhow::ensure!(!s.images.is_empty(), "[slideshow] has no images");
        }
        Ok(())
    }

    /// True if at least one output has a color image to show.
    pub fn has_wallpaper(&self) -> bool {
        self.wallpaper.is_some()
            || self.slideshow.is_some()
            || self.monitor.iter().any(|m| m.color.is_some())
    }

    fn override_for(&self, output_name: &str) -> Option<&MonitorOverride> {
        self.monitor.iter().find(|m| m.name == output_name)
    }

    /// True if the output shows the slideshow, rather than its own color
    /// or the global wallpaper.
    pub fn in_slideshow(&self, output_name: &str) -> bool {
        let own_color = self
            .override_for(output_name)
            .is_some_and(|m| m.color.is_some());
        self.slideshow.is_some() && !own_color
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

    #[test]
    fn transition_defaults_to_sweep_in() {
        let cfg = parse("");
        assert_eq!(cfg.daemon.transition, Transition::SweepIn);
        assert_eq!(cfg.daemon.transition_secs, 3.0);
        for (text, transition) in [
            ("sweep-in", Transition::SweepIn),
            ("sweep-out", Transition::SweepOut),
            ("morph", Transition::Morph),
            ("dissolve", Transition::Dissolve),
            ("portal", Transition::Portal),
            ("tide-in", Transition::TideIn),
            ("tide-out", Transition::TideOut),
        ] {
            let cfg = parse(&format!("[daemon]\ntransition = \"{text}\""));
            assert_eq!(cfg.daemon.transition, transition);
        }
    }

    #[cfg(feature = "clap")]
    #[test]
    fn transition_names_on_the_command_line_match_the_config() {
        use clap::ValueEnum;
        assert_eq!(Transition::value_variants(), Transition::ALL);
        for transition in Transition::ALL {
            let name = transition.to_possible_value().unwrap();
            assert_eq!(name.get_name(), transition.as_str());
        }
    }

    #[test]
    fn transition_strings_match_the_parser() {
        for transition in Transition::ALL {
            let cfg = parse(&format!(
                "[daemon]\ntransition = \"{}\"",
                transition.as_str()
            ));
            assert_eq!(cfg.daemon.transition, transition);
        }
    }

    #[test]
    fn unusable_transition_times_are_rejected() {
        for secs in ["-1.0", "nan", "inf", "1e30"] {
            let cfg = parse(&format!("[daemon]\ntransition_secs = {secs}"));
            assert!(cfg.validate().is_err(), "{secs}");
        }
        assert!(parse("[daemon]\ntransition_secs = 0.0").validate().is_ok());
    }

    #[test]
    fn transition_time_can_be_a_whole_number() {
        assert_eq!(
            parse("[daemon]\ntransition_secs = 5")
                .daemon
                .transition_secs,
            5.0
        );
    }

    #[test]
    fn slideshow_needs_images() {
        let cfg = parse("[slideshow]\nimages = []");
        assert!(cfg.validate().is_err());
        assert!(toml::from_str::<Config>("[slideshow]\ninterval_secs = 60").is_err());
    }

    #[test]
    fn slideshow_interval_has_a_default_and_cannot_be_zero() {
        let cfg = parse("[slideshow]\nimages = [\"/a.color.png\"]");
        assert_eq!(cfg.slideshow.unwrap().interval_secs.get(), 600);
        assert!(
            toml::from_str::<Config>("[slideshow]\nimages = [\"/a.png\"]\ninterval_secs = 0")
                .is_err()
        );
    }

    fn slideshow(text: &str) -> SlideshowConfig {
        parse(text).slideshow.unwrap()
    }

    fn at(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    #[test]
    fn slides_step_once_per_interval_and_wrap() {
        let s = slideshow("[slideshow]\nimages = [\"/a\", \"/b\", \"/c\"]\ninterval_secs = 60");
        let shown: Vec<usize> = [0, 59, 60, 150, 180].map(|t| s.due(at(t))).into();
        assert_eq!(shown, [0, 0, 1, 2, 0]);
    }

    #[test]
    fn shuffled_slides_show_each_image_once_per_round() {
        let s = slideshow(
            "[slideshow]\nimages = [\"/a\", \"/b\", \"/c\", \"/d\"]\ninterval_secs = 10\nshuffle = true",
        );
        assert!(!slideshow("[slideshow]\nimages = [\"/a\"]").shuffle);
        for round in 0..20 {
            let start = round * 40;
            let mut shown: Vec<usize> = (0..4).map(|i| s.due(at(start + i * 10))).collect();
            shown.sort();
            assert_eq!(shown, [0, 1, 2, 3], "round {round}");
        }
    }

    #[test]
    fn slides_infer_their_depth() {
        let cfg = parse("[slideshow]\nimages = [\"/a.color.png\", \"/b.jpg\"]");
        let slideshow = cfg.slideshow.unwrap();
        assert_eq!(
            slideshow.slide(0),
            (Path::new("/a.color.png"), PathBuf::from("/a.depth16.png"))
        );
        assert_eq!(slideshow.slide(1).1, PathBuf::from("/b.depth16.png"));
    }

    #[test]
    fn monitors_with_their_own_color_skip_the_slideshow() {
        let cfg = parse(
            r#"
            [slideshow]
            images = ["/a.color.png"]

            [[monitor]]
            name = "DP-1"
            color = "/mine.png"

            [[monitor]]
            name = "eDP-1"
            parallax_intensity = 0.1
            "#,
        );
        assert!(cfg.has_wallpaper());
        assert!(!cfg.in_slideshow("DP-1"));
        assert!(cfg.in_slideshow("eDP-1"));
        assert!(cfg.in_slideshow("HDMI-A-1"));
    }

    #[test]
    fn no_slideshow_by_default() {
        let cfg = parse("[wallpaper]\ncolor = \"/a.png\"");
        assert!(!cfg.in_slideshow("DP-1"));
    }
}
