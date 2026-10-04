use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Table};
use tracing::info;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    /// Optional so the CLI can parse a config that predates the first
    /// `fetch-model` or `set` call, or a daemon-only config with no
    /// [inference] section.
    #[serde(default)]
    pub inference: Option<InferenceConfig>,
    /// Optional so the CLI can still parse a fresh config file before
    /// the wallpaper section has been written by the first `set` call.
    #[serde(default)]
    pub wallpaper: Option<WallpaperConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)] // populated for future slideshow / inspection commands
pub struct WallpaperConfig {
    pub color: PathBuf,
    #[serde(default)]
    pub depth: Option<PathBuf>,
}

pub fn config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg).join("shiftpaper/config.toml")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config/shiftpaper/config.toml")
    } else {
        PathBuf::from("config.toml")
    }
}

/// Load the CLI's view of the shared config.
/// Returns Ok(None) if the file doesn't exist; Err if it exists but is malformed.
pub fn try_load() -> Result<Option<Config>> {
    let path = config_path();
    let Some(text) = read_text(&path)? else {
        return Ok(None);
    };

    let mut cfg: Config =
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;

    if let Some(ref mut i) = cfg.inference {
        i.model_path = expand_tilde(&i.model_path);
    }
    if let Some(ref mut w) = cfg.wallpaper {
        w.color = expand_tilde(&w.color);
        if let Some(ref mut d) = w.depth {
            *d = expand_tilde(d);
        }
    }

    Ok(Some(cfg))
}

/// Read config.toml as an editable document that keeps the user's
/// comments and formatting. A missing file reads as an empty document.
pub fn read_document() -> Result<DocumentMut> {
    read_document_at(&config_path())
}

/// Read config.toml, let `change` modify it, and write it back, creating
/// the file if needed. Nothing is written if `change` fails.
pub fn edit(change: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    edit_at(&config_path(), change)
}

/// The `[name]` table in `doc`, created if missing.
pub fn table<'a>(doc: &'a mut DocumentMut, name: &str) -> Result<&'a mut Table> {
    doc.entry(name)
        .or_insert(toml_edit::table())
        .as_table_mut()
        .with_context(|| format!("config [{name}] is not a table"))
}

fn read_document_at(path: &Path) -> Result<DocumentMut> {
    read_text(path)?
        .unwrap_or_default()
        .parse()
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn edit_at(path: &Path, change: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    let mut doc = read_document_at(path)?;
    change(&mut doc)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, doc.to_string())
        .with_context(|| format!("failed to write {}", path.display()))?;
    info!(path = %path.display(), "config updated");
    Ok(())
}

/// The file's contents, or None if it doesn't exist.
fn read_text(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("failed to read {}", path.display())),
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
    use toml_edit::value;

    fn set_hyprland_mode(doc: &mut DocumentMut) -> Result<()> {
        table(doc, "daemon")?["tracking_mode"] = value("hyprland");
        Ok(())
    }

    #[test]
    fn edit_creates_missing_file_and_table() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shiftpaper/config.toml");
        edit_at(&path, set_hyprland_mode).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "[daemon]\ntracking_mode = \"hyprland\"\n"
        );
    }

    #[test]
    fn edit_keeps_comments_and_other_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let before = "# my setup\n[daemon]\nparallax_intensity = 0.05 # subtle\ntracking_mode = \"pointer\"\n";
        std::fs::write(&path, before).unwrap();

        edit_at(&path, set_hyprland_mode).unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            before.replace("\"pointer\"", "\"hyprland\"")
        );
    }

    #[test]
    fn failed_edit_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        assert!(edit_at(&path, |_| anyhow::bail!("nope")).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn non_table_section_is_an_error() {
        let mut doc: DocumentMut = "daemon = 5".parse().unwrap();
        assert!(table(&mut doc, "daemon").is_err());
    }
}
