use anyhow::{Context, Result};
use shiftpaper_config::Config;

/// Load config.toml for the daemon, which has nothing to show until a
/// wallpaper has been set.
pub fn load() -> Result<Config> {
    let path = shiftpaper_config::path();
    let cfg = Config::load()?.with_context(|| {
        format!(
            "no config at {}, run `shiftpaper set <image>` first",
            path.display()
        )
    })?;
    anyhow::ensure!(
        cfg.has_wallpaper(),
        "no wallpaper set in {}, run `shiftpaper set <image>` first",
        path.display()
    );
    Ok(cfg)
}
