mod cache;
mod config;
mod depth;
mod fetch_model;
mod moge;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use shiftpaper_config::{Config, TrackingMode};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use toml_edit::{DocumentMut, value};
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Bake depth maps and configure the shiftpaperd parallax wallpaper daemon.
#[derive(Parser)]
#[command(
    name = "shiftpaper",
    version,
    about,
    long_about = "shiftpaper is the command-line companion to the shiftpaperd \
                  parallax wallpaper daemon. Use it to convert source images \
                  into the color + 16-bit depth pairs the daemon renders, set \
                  the active wallpaper, and switch cursor tracking modes.",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Bake a source image into a color + 16-bit depth PNG pair.
    ///
    /// Runs Depth Anything inference on the source image and writes a
    /// pair of files (`<hash>.color.png` and `<hash>.depth16.png`) to
    /// the cache directory or the directory specified by --out. Baking
    /// the same image again reuses the existing pair.
    Bake {
        /// Source image (jpeg, png, or webp).
        input: PathBuf,
        /// Output directory. Defaults to the shiftpaper cache directory
        /// at $XDG_CACHE_HOME/shiftpaper/wallpapers.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Path to the ONNX depth model. Falls back to
        /// $SHIFTPAPER_MODEL, then [inference] model_path in config.toml.
        #[arg(short, long, env = "SHIFTPAPER_MODEL")]
        model: Option<PathBuf>,
    },

    /// Bake an image and set it as the active wallpaper.
    ///
    /// Performs the same baking as `bake`, then points the daemon's
    /// config.toml at it, replacing any slideshow. Reload the daemon to
    /// show it. The resolved model path is also persisted to [inference]
    /// so future invocations don't need --model.
    Set {
        /// Source image (jpeg, png, or webp).
        input: PathBuf,
        /// Path to the ONNX depth model. Falls back to
        /// $SHIFTPAPER_MODEL, then [inference] model_path in config.toml.
        /// When provided, the resolved path is persisted to config.
        #[arg(short, long, env = "SHIFTPAPER_MODEL")]
        model: Option<PathBuf>,
    },

    /// Bake several images and show them in turn.
    ///
    /// Bakes each image like `set` does, then lists them in config.toml's
    /// [slideshow]. A folder adds the images directly inside it, in name
    /// order. Reload the daemon to start the slideshow. --stop, or `set`,
    /// goes back to a single wallpaper.
    Slideshow {
        /// Source images (jpeg, png, or webp), or folders of them.
        #[arg(required_unless_present = "stop")]
        inputs: Vec<PathBuf>,
        /// How long each image shows for, like 90s, 10m or 1h.
        #[arg(short, long, default_value = "10m", value_parser = parse_interval)]
        interval: NonZeroU32,
        /// Show the images in a random order, different each time round.
        #[arg(short, long)]
        shuffle: bool,
        /// Stop the slideshow, keeping the image it's showing.
        #[arg(long, conflicts_with_all = ["inputs", "interval", "shuffle"])]
        stop: bool,
        /// Path to the ONNX depth model. Falls back to
        /// $SHIFTPAPER_MODEL, then [inference] model_path in config.toml.
        #[arg(short, long, env = "SHIFTPAPER_MODEL")]
        model: Option<PathBuf>,
    },

    /// Download a depth model from HuggingFace and make it the one `set`
    /// and `bake` use.
    ///
    /// Without a name, downloads the default model. Files go to
    /// ~/.local/share/shiftpaper/models/<name>/ and the path is written to
    /// [inference] model_path in config.toml. Safe to re-run: the download
    /// is skipped if the files already exist, unless --force is given.
    /// Each model's weights have their own licence, shown before
    /// downloading.
    FetchModel {
        /// Which model to download. See --list.
        name: Option<String>,
        /// List the models that can be downloaded.
        #[arg(long, conflicts_with = "name")]
        list: bool,
        /// Re-download even if the files already exist.
        #[arg(long, short)]
        force: bool,
    },

    /// Show or change the cursor tracking mode.
    ///
    /// Pointer mode (the default) uses Wayland's native pointer events.
    /// It works on any wlr-layer-shell compositor, and the daemon only
    /// draws while the cursor is over visible desktop, which saves
    /// battery.
    ///
    /// Hyprland mode reads the global cursor position from the Hyprland
    /// IPC socket. Parallax remains responsive even when windows cover
    /// the desktop, at the cost of being Hyprland-specific. Note that
    /// Hyprland mode lets the daemon observe cursor positions over
    /// arbitrary windows, which is a minor privacy consideration.
    Mode {
        /// Tracking mode to set. Omit to print the current value.
        #[arg(value_enum)]
        mode: Option<TrackingMode>,
    },
}

fn main() -> Result<()> {
    // Logs go to stderr so stdout stays clean for the paths `bake` prints.
    // Log targets are named after the crate, which is the binary's name.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(concat!(env!("CARGO_CRATE_NAME"), "=warn"))),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Bake { input, out, model } => {
            let model = resolve_model(model)?;
            print_paths(&bake(&input, out.as_deref(), &model)?);
            Ok(())
        }
        Command::Set { input, model } => {
            let model = resolve_model(model)?;
            set(&input, &model)
        }
        Command::Slideshow { stop: true, .. } => stop_slideshow(),
        Command::Slideshow {
            inputs,
            interval,
            shuffle,
            model,
            ..
        } => {
            let model = resolve_model(model)?;
            slideshow(&inputs, interval, shuffle, &model)
        }
        Command::FetchModel { name, list, force } => fetch_model_cmd(name.as_deref(), list, force),
        Command::Mode { mode } => mode_cmd(mode),
    }
}

/// Resolve the model path. Clap merges --model and $SHIFTPAPER_MODEL into
/// `arg`, so by the time we get here a Some means flag-or-env. Falls through
/// to the config file, then the default fetch-model location, then errors.
fn resolve_model(arg: Option<PathBuf>) -> Result<PathBuf> {
    // 1. --model flag or $SHIFTPAPER_MODEL
    if let Some(p) = arg {
        return Ok(p);
    }
    // 2. [inference] model_path in config.toml
    if let Some(cfg) = Config::load()?
        && let Some(inference) = cfg.inference
    {
        let p = inference.model_path;
        if p.exists() {
            return Ok(p);
        }
        eprintln!(
            "warning: configured model path {} not found, checking default location",
            p.display()
        );
    }
    // 3. Default location written by `shiftpaper fetch-model`
    let default = fetch_model::default_model().path();
    if default.exists() {
        return Ok(default);
    }
    // 4. Friendly error
    anyhow::bail!(
        "no model found.\n\
         \n\
         Run `shiftpaper fetch-model` to download the default model, or\n\
         pass --model /path/to/model.onnx, or set $SHIFTPAPER_MODEL."
    )
}

fn bake(input: &Path, out: Option<&Path>, model: &Path) -> Result<cache::BakedPaths> {
    let rgba = image::open(input)
        .with_context(|| format!("failed to open {}", input.display()))?
        .to_rgba8();

    let hash = cache::hash_source(&rgba, model);
    let out_dir = out
        .map(|p| p.to_path_buf())
        .unwrap_or_else(cache::cache_dir);
    let paths = cache::paths_for(&hash, &out_dir);

    if cache::cache_hit(&paths) {
        info!("cache hit, skipping inference");
        return Ok(paths);
    }

    let depth_map = depth::estimate(&rgba, model)?;

    cache::write_color(&rgba, &paths.color)?;
    cache::write_depth(&depth_map, &paths.depth)?;

    info!("baked wallpaper");
    Ok(paths)
}

fn print_paths(paths: &cache::BakedPaths) {
    println!("{}", paths.color.display());
    println!("{}", paths.depth.display());
}

fn set(input: &Path, model: &Path) -> Result<()> {
    let paths = bake(input, None, model)?;
    print_paths(&paths);
    update_daemon_config(&paths, model)?;
    eprintln!();
    eprintln!("wallpaper set. reload the daemon to apply:");
    print_reload_hint();
    Ok(())
}

fn print_reload_hint() {
    eprintln!("  systemctl --user reload shiftpaperd");
    eprintln!("  # or: kill -HUP $(pidof shiftpaperd)");
}

fn slideshow(inputs: &[PathBuf], interval: NonZeroU32, shuffle: bool, model: &Path) -> Result<()> {
    let images = find_images(inputs)?;
    let mut baked = Vec::new();
    for (i, image) in images.iter().enumerate() {
        eprintln!("[{}/{}] {}", i + 1, images.len(), image.display());
        match bake(image, None, model) {
            Ok(paths) => baked.push(paths.color),
            // One bad image shouldn't stop the rest.
            Err(e) => eprintln!("  skipped: {e:#}"),
        }
    }
    anyhow::ensure!(!baked.is_empty(), "none of the images could be baked");

    update_slideshow_config(&baked, interval, shuffle, model)?;
    eprintln!();
    let plural = if baked.len() == 1 { "" } else { "s" };
    eprintln!(
        "slideshow of {} image{plural} set. reload the daemon to start it:",
        baked.len()
    );
    print_reload_hint();
    Ok(())
}

/// Swap the slideshow for a wallpaper of the image it's showing.
fn stop_slideshow() -> Result<()> {
    let Some(slideshow) = Config::load()?.and_then(|cfg| cfg.slideshow) else {
        eprintln!("there's no slideshow to stop");
        return Ok(());
    };
    let (color, depth) = slideshow.slide(slideshow.due(SystemTime::now()));
    config::edit(|doc| write_wallpaper(doc, color, &depth))?;
    eprintln!("slideshow stopped on the image it was showing. reload the daemon to apply:");
    print_reload_hint();
    Ok(())
}

/// The images to bake: files as given, and the images directly inside
/// any folders, in name order.
fn find_images(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    for input in inputs {
        if !input.is_dir() {
            images.push(input.clone());
            continue;
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(input)
            .with_context(|| format!("failed to read {}", input.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_file() && is_image(path))
            .collect();
        anyhow::ensure!(!found.is_empty(), "no images in {}", input.display());
        found.sort();
        images.append(&mut found);
    }
    Ok(images)
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            ["jpg", "jpeg", "png", "webp"]
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

/// Parse a duration like 90s, 10m or 1h into seconds. A bare number is
/// seconds.
fn parse_interval(text: &str) -> Result<NonZeroU32, String> {
    let (number, unit) = match text.find(|c: char| !c.is_ascii_digit()) {
        Some(i) => text.split_at(i),
        None => (text, "s"),
    };
    let scale = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => return Err(format!("`{unit}` isn't a unit. use s, m or h")),
    };
    number
        .parse::<u32>()
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .and_then(NonZeroU32::new)
        .ok_or_else(|| format!("`{text}` isn't a usable interval"))
}

fn mode_cmd(mode: Option<TrackingMode>) -> Result<()> {
    match mode {
        Some(m) => {
            update_tracking_mode(m)?;
            eprintln!("tracking mode set to {}", m.as_str());
            eprintln!();
            eprintln!("restart the daemon to apply:");
            eprintln!("  systemctl --user restart shiftpaperd");
        }
        None => {
            let mode = Config::load()?
                .map(|cfg| cfg.daemon.tracking_mode)
                .unwrap_or_default();
            println!("{}", mode.as_str());
        }
    }
    Ok(())
}

fn update_tracking_mode(mode: TrackingMode) -> Result<()> {
    config::edit(|doc| {
        config::table(doc, "daemon")?["tracking_mode"] = value(mode.as_str());
        Ok(())
    })
}

fn update_daemon_config(paths: &cache::BakedPaths, model: &Path) -> Result<()> {
    config::edit(|doc| {
        config::table(doc, "inference")?["model_path"] =
            value(model.to_string_lossy().into_owned());
        write_wallpaper(doc, &paths.color, &paths.depth)
    })
}

/// Point [wallpaper] at a baked pair, and remove any slideshow, which
/// would take its place.
fn write_wallpaper(doc: &mut DocumentMut, color: &Path, depth: &Path) -> Result<()> {
    let wallpaper = config::table(doc, "wallpaper")?;
    wallpaper["color"] = value(color.to_string_lossy().into_owned());
    wallpaper["depth"] = value(depth.to_string_lossy().into_owned());
    doc.remove("slideshow");
    Ok(())
}

fn update_slideshow_config(
    images: &[PathBuf],
    interval: NonZeroU32,
    shuffle: bool,
    model: &Path,
) -> Result<()> {
    config::edit(|doc| {
        config::table(doc, "inference")?["model_path"] =
            value(model.to_string_lossy().into_owned());
        let slideshow = config::table(doc, "slideshow")?;
        slideshow["images"] = value(config::list(images));
        slideshow["interval_secs"] = value(i64::from(interval.get()));
        slideshow["shuffle"] = value(shuffle);
        Ok(())
    })
}

fn fetch_model_cmd(name: Option<&str>, list: bool, force: bool) -> Result<()> {
    if list {
        fetch_model::print_list();
        return Ok(());
    }
    let model = match name {
        Some(name) => fetch_model::find(name)?,
        None => fetch_model::default_model(),
    };
    let path = fetch_model::fetch(model, force)?;
    fetch_model::persist_model_path(&path)?;
    eprintln!();
    eprintln!("model configured. you can now run:");
    eprintln!("  shiftpaper set ~/Pictures/wallpaper.jpg");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from([&["shiftpaper"], args].concat())
    }

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn slideshow_needs_images_unless_stopping() {
        assert!(parse(&["slideshow"]).is_err());
        assert!(parse(&["slideshow", "--stop"]).is_ok());
        assert!(parse(&["slideshow", "a.jpg", "--shuffle"]).is_ok());
    }

    #[test]
    fn stop_takes_nothing_else() {
        for extra in [&["a.jpg"][..], &["--shuffle"], &["-i", "1m"]] {
            let args = [&["slideshow", "--stop"][..], extra].concat();
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn intervals_take_units() {
        let secs = |text| parse_interval(text).map(NonZeroU32::get);
        assert_eq!(secs("90"), Ok(90));
        assert_eq!(secs("90s"), Ok(90));
        assert_eq!(secs("10m"), Ok(600));
        assert_eq!(secs("2h"), Ok(7200));
    }

    #[test]
    fn unusable_intervals_are_rejected() {
        for text in ["", "0", "0m", "m", "5d", "1.5h", "-1", "9999999h"] {
            assert!(parse_interval(text).is_err(), "{text}");
        }
    }

    #[test]
    fn folders_expand_to_their_images_in_order() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.PNG", "a.jpg", "notes.txt"] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        std::fs::create_dir(dir.path().join("sub.jpg")).unwrap();
        let single = PathBuf::from("/elsewhere/c.webp");

        let images = find_images(&[dir.path().to_path_buf(), single.clone()]).unwrap();
        assert_eq!(
            images,
            [dir.path().join("a.jpg"), dir.path().join("b.PNG"), single]
        );
    }

    #[test]
    fn a_folder_without_images_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_images(&[dir.path().to_path_buf()]).is_err());
    }
}
