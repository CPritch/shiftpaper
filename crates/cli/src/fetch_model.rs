use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use toml_edit::value;

/// A model `fetch-model` can download. Only permissively licensed models
/// are listed, so the default never puts anyone under extra terms. Any
/// other ONNX model can still be used with `--model`.
pub struct ModelSpec {
    /// Name used on the command line.
    pub name: &'static str,
    pub summary: &'static str,
    /// SPDX licence of the weights, as stated on the upstream model card.
    pub license: &'static str,
    /// The upstream model card the licence comes from.
    pub card: &'static str,
    /// HuggingFace repo with the ONNX export.
    pub repo: &'static str,
    /// Files to fetch from the repo. The first is the model itself, and any
    /// others (separate weights) are saved beside it under the same name,
    /// where ONNX Runtime expects them.
    pub files: &'static [&'static str],
    pub size_mb: u32,
}

pub const DEFAULT_MODEL: &str = "moge-2-vitb";

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "moge-2-vitb",
        summary: "MoGe-2 ViT-B. The most detail, and the slowest to bake.",
        license: "MIT",
        card: "https://huggingface.co/Ruicheng/moge-2-vitb-normal",
        repo: "Ruicheng/moge-2-vitb-normal-onnx",
        files: &["model.onnx"],
        size_mb: 419,
    },
    ModelSpec {
        name: "moge-2-vits",
        summary: "MoGe-2 ViT-S. Smaller and faster.",
        license: "MIT",
        card: "https://huggingface.co/Ruicheng/moge-2-vits-normal",
        repo: "Ruicheng/moge-2-vits-normal-onnx",
        files: &["model.onnx"],
        size_mb: 141,
    },
    ModelSpec {
        name: "depth-anything-v3-base",
        summary: "Depth Anything V3 Base.",
        license: "Apache-2.0",
        card: "https://huggingface.co/depth-anything/DA3-BASE",
        repo: "onnx-community/depth-anything-v3-base",
        files: &["onnx/model.onnx", "onnx/model.onnx_data"],
        size_mb: 413,
    },
    ModelSpec {
        name: "depth-anything-v3-small",
        summary: "Depth Anything V3 Small.",
        license: "Apache-2.0",
        card: "https://huggingface.co/depth-anything/DA3-SMALL",
        repo: "onnx-community/depth-anything-v3-small",
        files: &["onnx/model.onnx", "onnx/model.onnx_data"],
        size_mb: 105,
    },
    ModelSpec {
        name: "depth-anything-v2-small",
        summary: "Depth Anything V2 Small. The fastest, and the previous default.",
        license: "Apache-2.0",
        card: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small",
        repo: "onnx-community/depth-anything-v2-small",
        files: &["onnx/model.onnx"],
        size_mb: 99,
    },
];

/// Look a model up by name.
pub fn find(name: &str) -> Result<&'static ModelSpec> {
    MODELS.iter().find(|m| m.name == name).with_context(|| {
        format!("no model called `{name}`. `shiftpaper fetch-model --list` shows them")
    })
}

pub fn default_model() -> &'static ModelSpec {
    find(DEFAULT_MODEL).expect("the default model is in MODELS")
}

impl ModelSpec {
    fn dir(&self) -> PathBuf {
        models_dir().join(self.name)
    }

    /// Where the model file is once downloaded.
    pub fn path(&self) -> PathBuf {
        self.dir().join(file_name(self.files[0]))
    }

    fn is_downloaded(&self) -> bool {
        self.files
            .iter()
            .all(|f| self.dir().join(file_name(f)).exists())
    }
}

/// The last component of a path within a HuggingFace repo.
fn file_name(repo_path: &str) -> &str {
    repo_path.rsplit('/').next().unwrap_or(repo_path)
}

pub fn models_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "shiftpaper")
        .map(|d| d.data_local_dir().join("models"))
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".local/share/shiftpaper/models")
        })
}

pub fn print_list() {
    for m in MODELS {
        let mut name = m.name.to_string();
        if m.name == DEFAULT_MODEL {
            name.push_str(" (default)");
        }
        let downloaded = if m.is_downloaded() {
            "[downloaded]"
        } else {
            ""
        };
        let line = format!(
            "{name:<34}{:>4} MB  {:<11} {downloaded}",
            m.size_mb, m.license
        );
        println!("{}", line.trim_end());
        println!("    {}", m.summary);
    }
}

/// Download a model's files, unless they're already there. Returns the
/// path to the model file.
pub fn fetch(model: &ModelSpec, force: bool) -> Result<PathBuf> {
    eprintln!("{}: {}", model.name, model.summary);
    eprintln!("license: {} ({})", model.license, model.card);
    if model.is_downloaded() && !force {
        eprintln!("already downloaded to {}", model.dir().display());
        eprintln!("use --force to download it again");
        return Ok(model.path());
    }

    let dir = model.dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    for file in model.files {
        let url = format!("https://huggingface.co/{}/resolve/main/{file}", model.repo);
        download(&url, &dir.join(file_name(file)))?;
    }
    eprintln!("saved to {}", dir.display());
    Ok(model.path())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    eprintln!("downloading {url}");

    let response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP GET failed for {url}"))?;

    let content_length: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let pb = match content_length {
        Some(n) => {
            let pb = ProgressBar::new(n);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                     {bytes}/{total_bytes} ({eta})",
                )
                .expect("valid template")
                .progress_chars("#>-"),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {bytes} downloaded",
                )
                .expect("valid template"),
            );
            pb
        }
    };

    // Write to a .part file first so a failed download doesn't leave a
    // corrupt file that would pass the exists() check next time.
    let part_path = part_path(dest);
    {
        let mut reader = pb.wrap_read(response.into_body().into_reader());
        let mut file = std::fs::File::create(&part_path)
            .with_context(|| format!("failed to create {}", part_path.display()))?;
        std::io::copy(&mut reader, &mut file).context("download failed mid-transfer")?;
    }
    pb.finish();

    std::fs::rename(&part_path, dest)
        .with_context(|| format!("failed to finalise {}", dest.display()))
}

/// `model.onnx` downloads to `model.onnx.part`, `model.onnx_data` to
/// `model.onnx_data.part`, so two files of one model never share a name.
fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

pub fn persist_model_path(model_path: &Path) -> Result<()> {
    crate::config::edit(|doc| {
        crate::config::table(doc, "inference")?["model_path"] =
            value(model_path.to_string_lossy().into_owned());
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_listed() {
        assert_eq!(default_model().name, DEFAULT_MODEL);
    }

    #[test]
    fn model_names_are_unique() {
        for (i, m) in MODELS.iter().enumerate() {
            assert!(
                MODELS[i + 1..].iter().all(|other| other.name != m.name),
                "{} is listed twice",
                m.name
            );
        }
    }

    #[test]
    fn only_permissive_licenses_are_listed() {
        let permissive = [
            "MIT",
            "Apache-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "CC-BY-4.0",
        ];
        for m in MODELS {
            assert!(
                permissive.contains(&m.license),
                "{} is {}",
                m.name,
                m.license
            );
        }
    }

    #[test]
    fn the_model_file_comes_first() {
        for m in MODELS {
            assert!(m.files[0].ends_with(".onnx"), "{}", m.name);
        }
    }

    #[test]
    fn split_files_get_separate_part_files() {
        assert_eq!(
            part_path(Path::new("/m/model.onnx")),
            PathBuf::from("/m/model.onnx.part")
        );
        assert_eq!(
            part_path(Path::new("/m/model.onnx_data")),
            PathBuf::from("/m/model.onnx_data.part")
        );
    }
}
