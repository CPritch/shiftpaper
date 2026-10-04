use anyhow::{Context, Result};
use image::imageops::{self, FilterType};
use std::path::Path;

/// Depth map at the wallpaper's native resolution.
/// Values are u16 normalized, the GPU treats them as [0.0, 1.0] via R16Unorm.
pub struct DepthMap {
    pub data: Vec<u16>,
    pub width: u32,
    pub height: u32,
}

/// Load a 16-bit grayscale PNG produced by `shiftpaper-cli bake`.
pub fn load_depth_map(path: &Path) -> Result<DepthMap> {
    let img = image::open(path)
        .with_context(|| format!("failed to open depth map: {}", path.display()))?;
    let luma = img.to_luma16();
    let (width, height) = luma.dimensions();
    let data = luma.into_raw();

    Ok(DepthMap {
        data,
        width,
        height,
    })
}

/// How many points `DepthMap::ranks` samples.
pub const RANK_POINTS: usize = 257;

impl DepthMap {
    /// The map resized to `width` x `height`.
    pub fn resized(self, width: u32, height: u32) -> Self {
        let image =
            image::ImageBuffer::<image::Luma<u16>, _>::from_raw(self.width, self.height, self.data)
                .expect("a depth map's data matches its size");
        Self {
            data: imageops::resize(&image, width, height, FilterType::Triangle).into_raw(),
            width,
            height,
        }
    }

    /// The fraction of the map's pixels farther away than each of 257
    /// evenly spaced depths, from 0 (the farthest) to 1 (the nearest).
    /// Transitions sweep through this rank instead of raw depth, so they
    /// change a similar amount of the picture at every moment, however
    /// the scene's depths are spread out.
    pub fn ranks(&self) -> Vec<f32> {
        let mut counts = [0u32; RANK_POINTS - 1];
        for &d in &self.data {
            counts[usize::from(d >> 8)] += 1;
        }
        let total = self.data.len().max(1) as f32;
        let mut farther = 0;
        let mut ranks = Vec::with_capacity(RANK_POINTS);
        ranks.push(0.0);
        for n in counts {
            farther += n;
            ranks.push(farther as f32 / total);
        }
        ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(data: Vec<u16>) -> DepthMap {
        let width = data.len() as u32;
        DepthMap {
            data,
            width,
            height: 1,
        }
    }

    #[test]
    fn ranks_run_from_zero_to_one() {
        let ranks = map(vec![0, 1000, 30000, 65535]).ranks();
        assert_eq!(ranks.len(), RANK_POINTS);
        assert_eq!(ranks[0], 0.0);
        assert_eq!(ranks[RANK_POINTS - 1], 1.0);
        assert!(ranks.is_sorted());
    }

    #[test]
    fn ranks_follow_where_pixels_are() {
        // Three quarters of the pixels are far away, so the rank climbs to
        // 0.75 within the first band of depths.
        let ranks = map(vec![0, 0, 0, 65535]).ranks();
        assert_eq!(ranks[1], 0.75);
        assert_eq!(ranks[255], 0.75);
        assert_eq!(ranks[256], 1.0);
    }
}
