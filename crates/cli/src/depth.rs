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

/// MoGe resizes to its own token grid internally (about 1064 px wide for a
/// 16:10 image), so a larger input only costs time.
const MOGE_MAX_SIDE: u32 = 1536;

/// Run depth estimation on an already-loaded RGBA image.
/// Returns a normalized [0, 1] depth map at the source image's resolution
/// where 1.0 = closest to the camera.
pub fn estimate(rgba: &RgbaImage, model_path: &Path) -> Result<DepthMap> {
    let (orig_w, orig_h) = rgba.dimensions();
    #[cfg(feature = "load-dynamic")]
    check_onnxruntime_loads()?;
    let mut session = load_session(model_path)?;
    let model = ModelInput::read(&session)?;
    info!(w = orig_w, h = orig_h, family = ?model.family, "running depth estimation");

    let prediction = match model.family {
        Family::MoGe2 => run_moge(&mut session, rgba)?,
        Family::DepthAnythingV2 | Family::DepthAnythingV3 => {
            run_depth_anything(&mut session, &model, rgba)?
        }
    };
    let map =
        ImageBuffer::<Luma<f32>, _>::from_raw(prediction.width, prediction.height, prediction.data)
            .context("model output doesn't match its shape")?;

    Ok(DepthMap {
        data: imageops::resize(&map, orig_w, orig_h, FilterType::Triangle).into_raw(),
        width: orig_w,
        height: orig_h,
    })
}

/// Normalized disparity at the model's output resolution.
struct Prediction {
    data: Vec<f32>,
    width: u32,
    height: u32,
}

fn run_depth_anything(
    session: &mut Session,
    model: &ModelInput,
    rgba: &RgbaImage,
) -> Result<Prediction> {
    let (in_w, in_h) = model
        .fixed_size
        .unwrap_or_else(|| input_size(rgba.width(), rgba.height()));
    debug!(in_w, in_h, "model input size");

    let resized = imageops::resize(rgba, in_w, in_h, FilterType::Lanczos3);
    let (w, h) = (i64::from(in_w), i64::from(in_h));
    let shape = if model.family == Family::DepthAnythingV3 {
        vec![1, 1, 3, h, w]
    } else {
        vec![1, 3, h, w]
    };
    let pixels = to_input_tensor(&resized, MEAN, STD);
    let tensor = Tensor::from_array((shape, pixels.into_boxed_slice()))
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
    let output = if model.family == Family::DepthAnythingV3 {
        Output::Depth
    } else {
        Output::Disparity
    };
    Ok(Prediction {
        data: normalize_depth(raw, output),
        width: out_w as u32,
        height: out_h as u32,
    })
}

fn run_moge(session: &mut Session, rgba: &RgbaImage) -> Result<Prediction> {
    let scale = (MOGE_MAX_SIDE as f32 / rgba.width().max(rgba.height()) as f32).min(1.0);
    let in_w = (rgba.width() as f32 * scale).round() as u32;
    let in_h = (rgba.height() as f32 * scale).round() as u32;
    debug!(
        in_w,
        in_h,
        tokens = crate::moge::NUM_TOKENS,
        "model input size"
    );

    let resized = imageops::resize(rgba, in_w, in_h, FilterType::Lanczos3);
    // MoGe normalizes internally, so it takes plain [0, 1] RGB.
    let pixels = to_input_tensor(&resized, [0.0; 3], [1.0; 3]);
    let image = Tensor::from_array((
        vec![1, 3, i64::from(in_h), i64::from(in_w)],
        pixels.into_boxed_slice(),
    ))
    .map_err(|e| anyhow::anyhow!("failed to create image tensor: {e}"))?;
    // A scalar: an empty shape with one value.
    let tokens = Tensor::from_array((Vec::<i64>::new(), vec![crate::moge::NUM_TOKENS]))
        .map_err(|e| anyhow::anyhow!("failed to create token count tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs!["image" => image, "num_tokens" => tokens])
        .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;
    let (points_shape, points) = outputs["points"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract points: {e}"))?;
    let (_, mask) = outputs["mask"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract mask: {e}"))?;

    let [_, out_h, out_w, 3] = **points_shape else {
        anyhow::bail!("unexpected points shape {:?}", &**points_shape);
    };
    let depth = crate::moge::depth_from_points(points, mask, out_w as usize, out_h as usize);
    Ok(Prediction {
        data: normalize_depth(&depth, Output::Depth),
        width: out_w as u32,
        height: out_h as u32,
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

/// Load the model, preferring CUDA. If CUDA isn't available, ort logs it
/// and falls back to the CPU on its own. AMD's MIGraphX provider isn't
/// used: as of MIGraphX 7.2 it aborts while compiling Depth Anything.
fn load_session(model_path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow::anyhow!("failed to create ONNX session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .map_err(|e| anyhow::anyhow!("failed to set execution providers: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to load ONNX model from {}: {e}",
                model_path.display()
            )
        })
}

/// The kinds of model we know how to run, recognised from their inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    /// One 4-D image input, disparity out: Depth Anything V2 and models
    /// exported the same way.
    DepthAnythingV2,
    /// One 5-D input ([batch, views, C, H, W]), distance out.
    DepthAnythingV3,
    /// An image and a token count in, a 3D point per pixel out.
    MoGe2,
}

/// What the model expects as input, read from its metadata.
struct ModelInput {
    family: Family,
    /// Width and height, if the model was exported with a fixed size.
    fixed_size: Option<(u32, u32)>,
}

impl ModelInput {
    fn read(session: &Session) -> Result<Self> {
        let inputs = session.inputs();
        let dims: &[i64] = inputs
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .context("model has no tensor input")?;
        let [.., h, w] = *dims else {
            anyhow::bail!("unsupported model input shape {dims:?}");
        };
        let family = if inputs.iter().any(|input| input.name() == "num_tokens") {
            Family::MoGe2
        } else {
            match dims.len() {
                4 => Family::DepthAnythingV2,
                5 => Family::DepthAnythingV3,
                _ => anyhow::bail!("unsupported model input shape {dims:?}"),
            }
        };
        // Dynamic dimensions are reported as -1.
        let fixed_size = (h > 0 && w > 0).then_some((w as u32, h as u32));
        Ok(Self { family, fixed_size })
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

/// The image as separate R, G and B planes, scaled to [0, 1] and then
/// normalized with `mean` and `std`, which is the layout the models expect.
fn to_input_tensor(image: &RgbaImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let n = image.pixels().len();
    let mut out = vec![0.0f32; 3 * n];
    for (i, px) in image.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    out
}

/// What a model's output values mean.
#[derive(Clone, Copy)]
enum Output {
    /// Higher is nearer, already proportional to parallax (Depth Anything V2).
    Disparity,
    /// Distance from the camera (Depth Anything V3, MoGe-2).
    Depth,
}

/// Rescale raw model output to [0, 1] disparity, with 1.0 nearest the
/// camera. Depth is converted to disparity first, because parallax shift is
/// proportional to 1 / distance; flipping depth linearly squashes the
/// foreground together.
fn normalize_depth(raw: &[f32], output: Output) -> Vec<f32> {
    let disparity: Vec<f32> = match output {
        Output::Disparity => raw.to_vec(),
        Output::Depth => raw.iter().map(|&v| 1.0 / v.max(1e-6)).collect(),
    };
    let (min, max) = disparity
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (max - min).max(1e-6);
    disparity.iter().map(|&v| (v - min) / range).collect()
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
        let tensor = to_input_tensor(&image, MEAN, STD);
        let red = (1.0 - MEAN[0]) / STD[0];
        let green = (1.0 - MEAN[1]) / STD[1];
        // [R plane, G plane, B plane], each two pixels long.
        assert_eq!(tensor[0], red);
        assert_eq!(tensor[3], green);
    }

    #[test]
    fn disparity_is_normalized() {
        assert_eq!(
            normalize_depth(&[2.0, 4.0, 6.0], Output::Disparity),
            [0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn depth_becomes_disparity() {
        // Distances 1, 2 and 4 are disparities 1, 0.5 and 0.25.
        let d = normalize_depth(&[1.0, 2.0, 4.0], Output::Depth);
        assert_eq!(d[0], 1.0);
        assert!((d[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(d[2], 0.0);
    }

    #[test]
    fn flat_depth_does_not_divide_by_zero() {
        assert_eq!(normalize_depth(&[3.0, 3.0], Output::Disparity), [0.0, 0.0]);
        assert_eq!(normalize_depth(&[0.0, 0.0], Output::Depth), [0.0, 0.0]);
    }
}
