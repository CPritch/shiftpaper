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

    /// Roughly the fraction of the map's pixels farther away than each of
    /// 257 evenly spaced depths, from 0 (the farthest) to 1 (the nearest).
    /// Transitions sweep through this rank instead of raw depth, so they
    /// change a similar amount of the picture at every moment, however the
    /// scene's depths are spread out. See `cumulative` for the rough part.
    pub fn ranks(&self) -> Vec<f32> {
        let mut counts = [0u32; RANK_POINTS - 1];
        for &d in &self.data {
            counts[usize::from(d >> 8)] += 1;
        }
        cumulative(&counts)
    }

    /// Like `ranks`, but for each pixel's rough height in the scene from
    /// `tide_height`: the fraction of pixels lower than each of 257 evenly
    /// spaced heights from -1 to 1.
    pub fn height_ranks(&self) -> Vec<f32> {
        let mut counts = [0u32; RANK_POINTS - 1];
        let width = self.width.max(1) as usize;
        for (i, &d) in self.data.iter().enumerate() {
            let y = ((i / width) as f32 + 0.5) / self.height as f32;
            let height = tide_height(y, f32::from(d) / 65535.0);
            let bin = ((height + 1.0) * 0.5 * (RANK_POINTS - 1) as f32) as usize;
            counts[bin.min(RANK_POINTS - 2)] += 1;
        }
        cumulative(&counts)
    }
}

/// How many cells across and down `DepthMap::grid` divides a map into.
pub const GRID_CELLS: usize = 32;

/// The range of depths in each cell of a coarse grid over a depth map, so
/// the CPU can tell roughly where things are in the scene without reading
/// the map back from the GPU.
#[derive(Clone)]
pub struct DepthGrid {
    /// GRID_CELLS rows of GRID_CELLS cells.
    pub cells: Vec<DepthCell>,
}

/// Depths from 0 to 1 in one cell of a `DepthGrid`.
#[derive(Clone, Copy)]
pub struct DepthCell {
    pub least: f32,
    pub greatest: f32,
    pub mean: f32,
}

impl DepthGrid {
    /// For a wallpaper without a depth map, which the GPU sees as all 0.
    pub fn flat() -> Self {
        let cell = DepthCell {
            least: 0.0,
            greatest: 0.0,
            mean: 0.0,
        };
        Self {
            cells: vec![cell; GRID_CELLS * GRID_CELLS],
        }
    }
}

impl DepthMap {
    /// The map's `DepthGrid`. A pixel at (x, y) is in the cell
    /// `x * GRID_CELLS / width` across and `y * GRID_CELLS / height` down.
    pub fn grid(&self) -> DepthGrid {
        // The least, greatest and total depth in each cell, and how many
        // pixels it has.
        let mut sums = vec![(f32::MAX, f32::MIN, 0.0, 0u32); GRID_CELLS * GRID_CELLS];
        let width = self.width.max(1) as usize;
        for (i, &d) in self.data.iter().enumerate() {
            let col = (i % width) * GRID_CELLS / width;
            let row = (i / width) * GRID_CELLS / self.height as usize;
            let (least, greatest, total, count) = &mut sums[row * GRID_CELLS + col];
            let d = f32::from(d) / 65535.0;
            *least = least.min(d);
            *greatest = greatest.max(d);
            *total += d;
            *count += 1;
        }
        let cells = sums
            .into_iter()
            .map(|(least, greatest, total, count)| match count {
                // A map smaller than the grid leaves some cells empty, so
                // they allow any depth.
                0 => DepthCell {
                    least: 0.0,
                    greatest: 1.0,
                    mean: 0.5,
                },
                _ => DepthCell {
                    least,
                    greatest,
                    mean: total / count as f32,
                },
            })
            .collect();
        DepthGrid { cells }
    }
}

/// Where the tide takes the horizon to be, as a fraction of the way down
/// the image. Must match HORIZON in shader.wgsl.
pub const HORIZON: f32 = 0.5;
/// How close a depth of 0 is taken to be, which keeps the farthest pixels
/// at a finite distance. Must match TIDE_NEAR in shader.wgsl.
pub const TIDE_NEAR: f32 = 0.1;

/// Roughly how high a point is in the scene, squashed into (-1, 1), from
/// how far down the image it is and its depth. Distance comes from depth,
/// and height from how far above or below the horizon the point is at
/// that distance. The model's depth has no true scale, so this is only a
/// guess, but a good enough one for water to find its level. Must match
/// tide_height in shader.wgsl.
pub fn tide_height(y: f32, depth: f32) -> f32 {
    let distance = 1.0 / (depth + TIDE_NEAR);
    let height = (HORIZON - y) * distance;
    height / (1.0 + height.abs())
}

/// The most of a transition any one slice of depth or height can take, as
/// a fraction of the picture.
const SLICE_CAP: f32 = 0.02;

/// Running totals of `counts` as fractions of the whole, starting from 0.
/// Each slice counts for no more than SLICE_CAP of the picture. A big area
/// all at one depth, like a sky the model marks as infinitely far, would
/// otherwise get a share of the transition to match its size, but it all
/// switches at the same moment, so the rest of that share is a pause.
fn cumulative(counts: &[u32]) -> Vec<f32> {
    let cap = ((counts.iter().sum::<u32>() as f32 * SLICE_CAP).ceil() as u32).max(1);
    let capped: Vec<u32> = counts.iter().map(|&n| n.min(cap)).collect();
    let total = capped.iter().sum::<u32>().max(1) as f32;
    let mut sum = 0;
    let mut ranks = Vec::with_capacity(counts.len() + 1);
    ranks.push(0.0);
    for n in capped {
        sum += n;
        ranks.push(sum as f32 / total);
    }
    ranks
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
    fn lower_and_nearer_is_lower_down() {
        // At the same depth, lower in the image is lower in the scene.
        assert!(tide_height(0.9, 0.5) < tide_height(0.6, 0.5));
        // Far away and high in the image, like sky, is highest of all.
        assert!(tide_height(0.1, 0.0) > tide_height(0.3, 0.5));
        assert!(tide_height(0.1, 0.0) < 1.0 && tide_height(1.0, 0.0) > -1.0);
    }

    #[test]
    fn height_ranks_run_from_zero_to_one() {
        let map = DepthMap {
            data: vec![0, 20000, 40000, 65535, 0, 20000, 40000, 65535],
            width: 4,
            height: 2,
        };
        let ranks = map.height_ranks();
        assert_eq!(ranks.len(), RANK_POINTS);
        assert_eq!(ranks[0], 0.0);
        assert_eq!(ranks[RANK_POINTS - 1], 1.0);
        assert!(ranks.is_sorted());
    }

    #[test]
    fn ranks_follow_where_pixels_are() {
        // 100 pixels spread evenly over 100 depths, so each takes 1%.
        let ranks = map((0..100).map(|i| i * 650).collect()).ranks();
        assert!((ranks[128] - 0.51).abs() < 0.02, "{}", ranks[128]);
    }

    #[test]
    fn one_depth_takes_only_a_moment() {
        // A sky at depth 0 covers half the picture, and the other half is
        // spread over 100 depths. The sky still all switches together, but
        // takes a sliver of the transition rather than half of it, so the
        // sweep doesn't idle through it.
        let mut data = vec![0; 100];
        data.extend((0..100).map(|i| 1000 + i * 640));
        let ranks = map(data).ranks();
        let sky = ranks[1] - ranks[0];
        assert!(sky < 0.05, "the sky takes {sky}");
    }
}
