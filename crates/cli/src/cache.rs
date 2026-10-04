use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::depth::DepthMap;

/// Paths for a baked wallpaper pair.
pub struct BakedPaths {
    pub color: PathBuf,
    pub depth: PathBuf,
}

/// Default cache directory: `~/.cache/shiftpaper/wallpapers/`.
pub fn cache_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "shiftpaper")
        .map(|dirs| dirs.cache_dir().join("wallpapers"))
        .unwrap_or_else(|| PathBuf::from("/tmp/shiftpaper-cache/wallpapers"))
}

/// Bump when a change to baking alters its output, so images baked by an
/// older version are baked again instead of reused from the cache.
const BAKE_VERSION: &[u8] = b"3";

/// Cache key for a source image baked with a model: blake3 of the bake
/// version, the model's path and size, and the decoded RGBA bytes. The same
/// image baked with a different model gets a different key.
pub fn hash_source(rgba: &image::RgbaImage, model: &Path) -> String {
    let model = model.canonicalize().unwrap_or_else(|_| model.to_path_buf());
    let model_size = std::fs::metadata(&model).map_or(0, |m| m.len());
    blake3::Hasher::new()
        .update(BAKE_VERSION)
        .update(model.as_os_str().as_encoded_bytes())
        .update(&model_size.to_le_bytes())
        .update(rgba.as_raw())
        .finalize()
        .to_hex()
        .to_string()
}

/// Compute the color+depth paths for a given hash in a given directory.
pub fn paths_for(hash: &str, base: &Path) -> BakedPaths {
    BakedPaths {
        color: base.join(format!("{hash}.color.png")),
        depth: base.join(format!("{hash}.depth16.png")),
    }
}

/// True if both files exist.
pub fn cache_hit(paths: &BakedPaths) -> bool {
    paths.color.exists() && paths.depth.exists()
}

/// Write the RGBA source as a PNG at the target path.
pub fn write_color(rgba: &image::RgbaImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    rgba.save(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Write a normalized f32 depth map as a 16-bit grayscale PNG.
pub fn write_depth(depth: &DepthMap, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let pixels: Vec<u16> = depth
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
        .collect();

    let buffer =
        image::ImageBuffer::<image::Luma<u16>, _>::from_raw(depth.width, depth.height, pixels)
            .context("failed to build depth image buffer")?;

    buffer
        .save(path)
        .with_context(|| format!("failed to write {}", path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_depends_on_the_model() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.onnx"), dir.path().join("b.onnx"));
        std::fs::write(&a, b"model a").unwrap();
        std::fs::write(&b, b"model b").unwrap();
        let image = image::RgbaImage::new(2, 2);

        assert_eq!(hash_source(&image, &a), hash_source(&image, &a));
        assert_ne!(hash_source(&image, &a), hash_source(&image, &b));
    }
}
