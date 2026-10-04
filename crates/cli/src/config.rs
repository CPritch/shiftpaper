//! Editing config.toml in place. Reading it typed goes through
//! `shiftpaper_config::Config::load`.

use anyhow::{Context, Result};
use std::path::Path;
use toml_edit::{DocumentMut, Table};
use tracing::info;

/// Read config.toml, let `change` modify it, and write it back, keeping
/// the user's comments and formatting. Creates the file if needed, and
/// writes nothing if `change` fails.
pub fn edit(change: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    edit_at(&shiftpaper_config::path(), change)
}

/// The `[name]` table in `doc`, created if missing.
pub fn table<'a>(doc: &'a mut DocumentMut, name: &str) -> Result<&'a mut Table> {
    doc.entry(name)
        .or_insert(toml_edit::table())
        .as_table_mut()
        .with_context(|| format!("config [{name}] is not a table"))
}

fn edit_at(path: &Path, change: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
    let mut doc: DocumentMut = shiftpaper_config::read_text(path)?
        .unwrap_or_default()
        .parse()
        .with_context(|| format!("failed to parse {}", path.display()))?;
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
