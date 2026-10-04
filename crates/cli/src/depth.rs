use anyhow::{Context, Result};
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Luma, RgbaImage};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tracing::{debug, info};

pub struct DepthMap {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Depth Anything's preferred length for the image's shorter side.
const SHORT_SIDE: u32 = 518;
/// The model's patch size. Input sides must be a multiple of it.
const PATCH: u32 = 14;

// ImageNet normalization constants used by Depth Anything V2/V3
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Run depth estimation on an already-loaded RGBA image.
/// Returns a normalized [0, 1] depth map at the source image's resolution
/// where 1.0 = closest to the camera.
pub fn estimate(rgba: &RgbaImage, model_path: &Path) -> Result<DepthMap> {
    let (orig_w, orig_h) = rgba.dimensions();
    #[cfg(feature = "load-dynamic")]
    check_onnxruntime_loads()?;
    let mut session = load_session(model_path)?;
    let model = ModelInput::read(&session)?;
    let (in_w, in_h) = model
        .fixed_size
        .unwrap_or_else(|| input_size(orig_w, orig_h));
    info!(
        w = orig_w,
        h = orig_h,
        in_w,
        in_h,
        "running depth estimation"
    );
    debug!(rank = model.rank, "model input");

    let resized = imageops::resize(rgba, in_w, in_h, FilterType::Lanczos3);
    let (w, h) = (i64::from(in_w), i64::from(in_h));
    let shape = if model.rank == 5 {
        vec![1, 1, 3, h, w]
    } else {
        vec![1, 3, h, w]
    };
    let tensor = Tensor::from_array((shape, to_input_tensor(&resized).into_boxed_slice()))
        .map_err(|e| anyhow::anyhow!("failed to create input tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;
    let (out_shape, raw) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract output: {e}"))?;

    // The output ends in [height, width], whatever leading dimensions it has.
    let [.., out_h, out_w] = **out_shape else {
        anyhow::bail!("unexpected output shape {:?}", &**out_shape);
    };
    // V2 predicts inverse depth (higher is nearer); V3 predicts direct depth.
    let depth = normalize_depth(raw, model.rank == 5);
    let map = ImageBuffer::<Luma<f32>, _>::from_raw(out_w as u32, out_h as u32, depth)
        .with_context(|| format!("output shape {:?} doesn't match its data", &**out_shape))?;

    Ok(DepthMap {
        data: imageops::resize(&map, orig_w, orig_h, FilterType::Triangle).into_raw(),
        width: orig_w,
        height: orig_h,
    })
}

/// Check that libonnxruntime loads before ort tries to. When ort's own load
/// fails, it builds the error through the library it couldn't load and
/// deadlocks, so a broken install would hang with no output.
#[cfg(feature = "load-dynamic")]
fn check_onnxruntime_loads() -> Result<()> {
    // The same lookup as ort: $ORT_DYLIB_PATH or libonnxruntime.so, using
    // a copy beside the executable if there is one.
    let name = std::env::var_os("ORT_DYLIB_PATH")
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "libonnxruntime.so".into());
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join(&name)))
        .filter(|p| p.exists());
    let path = beside_exe.unwrap_or_else(|| name.into());

    // SAFETY: loading a library runs its initialisers, and this is the
    // library ort is about to load anyway.
    let lib = unsafe { libloading::Library::new(&path) }.with_context(|| {
        format!(
            "failed to load ONNX Runtime from {}. Is onnxruntime installed, \
             with all of its dependencies up to date?",
            path.display()
        )
    })?;
    // Keep it loaded so ort's own load reuses it instead of loading it again.
    std::mem::forget(lib);
    Ok(())
}

/// Load the model, preferring a GPU: CUDA for NVIDIA, MIGraphX for AMD.
/// Providers missing from the installed onnxruntime are skipped, and ort
/// falls back to the CPU on its own.
fn load_session(model_path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow::anyhow!("failed to create ONNX session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
        .with_execution_providers([
            ort::ep::CUDA::default().build(),
            ort::ep::MIGraphX::default().build(),
        ])
        .map_err(|e| anyhow::anyhow!("failed to set execution providers: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to load ONNX model from {}: {e}",
                model_path.display()
            )
        })
}

/// What the model expects as input, read from its metadata.
struct ModelInput {
    /// 5 for Depth Anything V3 ([batch, views, C, H, W]), 4 for V2.
    rank: usize,
    /// Width and height, if the model was exported with a fixed size.
    fixed_size: Option<(u32, u32)>,
}

impl ModelInput {
    fn read(session: &Session) -> Result<Self> {
        let dims: &[i64] = session
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .context("model has no tensor input")?;
        let [.., h, w] = *dims else {
            anyhow::bail!("unsupported model input shape {dims:?}");
        };
        anyhow::ensure!(
            matches!(dims.len(), 4 | 5),
            "unsupported model input shape {dims:?}"
        );
        // Dynamic dimensions are reported as -1.
        let fixed_size = (h > 0 && w > 0).then_some((w as u32, h as u32));
        Ok(Self {
            rank: dims.len(),
            fixed_size,
        })
    }
}

/// Model input size for an image: the shorter side scaled to 518 and both
/// sides rounded to a multiple of 14, keeping the aspect ratio. This
/// matches Depth Anything's own preprocessing.
fn input_size(width: u32, height: u32) -> (u32, u32) {
    let scale = SHORT_SIDE as f32 / width.min(height) as f32;
    let fit = |side: u32| {
        let patches = (side as f32 * scale / PATCH as f32).round() as u32;
        (patches * PATCH).max(SHORT_SIDE)
    };
    (fit(width), fit(height))
}

/// The image as separate R, G and B planes with ImageNet normalization,
/// which is the layout the model expects.
fn to_input_tensor(image: &RgbaImage) -> Vec<f32> {
    let n = image.pixels().len();
    let mut out = vec![0.0f32; 3 * n];
    for (i, px) in image.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// Rescale raw model output to [0, 1], flipping it if `invert` so that 1.0
/// is always nearest the camera.
fn normalize_depth(raw: &[f32], invert: bool) -> Vec<f32> {
    let (min, max) = raw
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (max - min).max(1e-6);
    raw.iter()
        .map(|&v| {
            let t = (v - min) / range;
            if invert { 1.0 - t } else { t }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_keeps_aspect_ratio() {
        assert_eq!(input_size(1875, 1116), (868, 518));
        assert_eq!(input_size(2560, 1440), (924, 518));
        assert_eq!(input_size(1440, 2560), (518, 924));
    }

    #[test]
    fn input_is_at_least_518_on_each_side() {
        assert_eq!(input_size(1000, 1000), (518, 518));
        assert_eq!(input_size(400, 300), (686, 518));
    }

    #[test]
    fn input_tensor_is_planar() {
        let image = RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]).unwrap();
        let tensor = to_input_tensor(&image);
        let red = (1.0 - MEAN[0]) / STD[0];
        let green = (1.0 - MEAN[1]) / STD[1];
        // [R plane, G plane, B plane], each two pixels long.
        assert_eq!(tensor[0], red);
        assert_eq!(tensor[3], green);
    }

    #[test]
    fn depth_is_normalized_and_optionally_inverted() {
        assert_eq!(normalize_depth(&[2.0, 4.0, 6.0], false), [0.0, 0.5, 1.0]);
        assert_eq!(normalize_depth(&[2.0, 4.0, 6.0], true), [1.0, 0.5, 0.0]);
    }

    #[test]
    fn flat_depth_does_not_divide_by_zero() {
        assert_eq!(normalize_depth(&[3.0, 3.0], false), [0.0, 0.0]);
    }
}
