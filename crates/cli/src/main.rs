mod cache;
mod config;
mod depth;
mod fetch_model;
mod moge;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use shiftpaper_config::{Config, TrackingMode};
use std::path::{Path, PathBuf};
use toml_edit::value;
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
        /// Path to the Depth Anything ONNX model. Falls back to
        /// $SHIFTPAPER_MODEL, then [inference] model_path in config.toml.
        #[arg(short, long, env = "SHIFTPAPER_MODEL")]
        model: Option<PathBuf>,
    },

    /// Bake an image and set it as the active wallpaper.
    ///
    /// Performs the same baking as `bake`, then points the daemon's
    /// config.toml at it. Reload the daemon to show it. The resolved
    /// model path is also persisted to [inference] so future
    /// invocations don't need --model.
    Set {
        /// Source image (jpeg, png, or webp).
        input: PathBuf,
        /// Path to the Depth Anything ONNX model. Falls back to
        /// $SHIFTPAPER_MODEL, then [inference] model_path in config.toml.
        /// When provided, the resolved path is persisted to config.
        #[arg(short, long, env = "SHIFTPAPER_MODEL")]
        model: Option<PathBuf>,
    },

    /// Download the default depth model from HuggingFace.
    ///
    /// Downloads Depth Anything V2 Small (ONNX) from the onnx-community
    /// repository to ~/.local/share/shiftpaper/models/ and writes the path
    /// to [inference] model_path in config.toml. Safe to re-run: the
    /// download is skipped if the file already exists, unless --force
    /// is given.
    FetchModel {
        /// Override the download URL. Defaults to the onnx-community
        /// release on HuggingFace.
        #[arg(long)]
        url: Option<String>,
        /// Re-download even if the file already exists.
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
            bake(&input, out.as_deref(), &model)?;
            Ok(())
        }
        Command::Set { input, model } => {
            let model = resolve_model(model)?;
            set(&input, &model)
        }
        Command::FetchModel { url, force } => fetch_model_cmd(url.as_deref(), force),
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
    let default = fetch_model::default_model_path();
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
        println!("{}", paths.color.display());
        println!("{}", paths.depth.display());
        return Ok(paths);
    }

    let depth_map = depth::estimate(&rgba, model)?;

    cache::write_color(&rgba, &paths.color)?;
    cache::write_depth(&depth_map, &paths.depth)?;

    info!("baked wallpaper");
    println!("{}", paths.color.display());
    println!("{}", paths.depth.display());

    Ok(paths)
}

fn set(input: &Path, model: &Path) -> Result<()> {
    let paths = bake(input, None, model)?;
    update_daemon_config(&paths, model)?;
    eprintln!();
    eprintln!("wallpaper set. reload the daemon to apply:");
    eprintln!("  systemctl --user reload shiftpaperd");
    eprintln!("  # or: kill -HUP $(pidof shiftpaperd)");
    Ok(())
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
        let wallpaper = config::table(doc, "wallpaper")?;
        wallpaper["color"] = value(paths.color.to_string_lossy().into_owned());
        wallpaper["depth"] = value(paths.depth.to_string_lossy().into_owned());
        Ok(())
    })
}

fn fetch_model_cmd(url: Option<&str>, force: bool) -> Result<()> {
    let dest = fetch_model::default_model_path();
    let url = url.unwrap_or(fetch_model::DEFAULT_MODEL_URL);
    fetch_model::fetch_model(url, &dest, force)?;
    fetch_model::persist_model_path(&dest)?;
    eprintln!();
    eprintln!("model configured. you can now run:");
    eprintln!("  shiftpaper set ~/Pictures/wallpaper.jpg");
    Ok(())
}
