use crate::depth::{
    DepthCell, DepthGrid, DepthMap, GRID_CELLS, Ground, RANK_POINTS, load_depth_map,
};
use anyhow::{Context, Result};
use bytemuck::Zeroable;
use image::imageops::{self, FilterType};
use shiftpaper_config::Transition;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};
use wgpu::CurrentSurfaceTexture;
use wgpu::util::DeviceExt;

/// Mirrors `Uniforms` in shader.wgsl. `repr(C)` fixes the field order and
/// padding to match the shader, and `Pod` lets bytemuck view it as bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cursor_offset: [f32; 2],
    intensity: f32,
    progress: f32,
    uv_scale: [f32; 2],
    next_uv_scale: [f32; 2],
    uv_center: [f32; 2],
    next_uv_center: [f32; 2],
    /// From `shader_style`.
    style: u32,
    /// The screen's width over its height.
    aspect: f32,
    /// Where the spheres of a portal or dissolve grow from: how many points
    /// are in use, how far a sphere grows, the points themselves, how the
    /// spheres are paced and when they show, from `Trail`.
    trail_len: u32,
    portal_reach: f32,
    trail: [[f32; 4]; TRAIL_POINTS],
    portal_pace: [[f32; 4]; PACE_POINTS.div_ceil(4)],
    portal_timing: [[f32; 4]; PACE_POINTS.div_ceil(4)],
    /// Where the ground is in each wallpaper, for the tide.
    ground: [f32; 4],
    next_ground: [f32; 4],
    /// The effect playing, from `shader_effect`, or 0 for none, how far
    /// through it is, from 0 to 1, and where it starts from, in screen uv.
    effect: u32,
    effect_progress: f32,
    effect_origin: [f32; 2],
}

/// How many points a trail can hold. Must match TRAIL_POINTS in
/// shader.wgsl.
const TRAIL_POINTS: usize = 32;
/// How far the cursor moves, in screen heights, before the portal grows
/// from a new point.
const TRAIL_SPACING: f32 = 0.05;
/// How close a depth of 0 is taken to be, which keeps the farthest pixels
/// at a finite distance, as with the tide's TIDE_NEAR. Must match
/// PORTAL_NEAR in shader.wgsl.
const PORTAL_NEAR: f32 = 0.1;
/// How many distances `Trail::pace` covers, and times `Trail::timing`
/// does. Must match PACE_POINTS in shader.wgsl.
const PACE_POINTS: usize = 33;
/// How soft the edge of a sphere is, as a fraction of the transition.
/// Must match PORTAL_EDGE in shader.wgsl.
const PORTAL_EDGE: f32 = 0.015;
/// How many spheres a dissolve grows, at most.
const DISSOLVE_SPHERES: usize = 16;
/// How far through a dissolve its last sphere starts.
const DISSOLVE_SPAWNING: f32 = 0.7;
/// The spheres in a dissolve start at this power of how far through
/// DISSOLVE_SPAWNING they are, so below 1 they start more often as it goes.
const DISSOLVE_QUICKENING: f32 = 0.8;
/// How many times likelier a dissolve's spheres are to start among the
/// furthest things than among the nearest.
const DISSOLVE_FAR: f32 = 20.0;

/// Where the spheres of a portal or dissolve grow from. The new wallpaper
/// grows from each point like a sphere, from its start: for the portal,
/// from when the cursor got there.
#[derive(Clone, Copy)]
struct Trail {
    /// x and y in screen uv, then the transition's progress when the
    /// sphere starts. The fourth number pads to the shader's vec4. They're
    /// in the order they start.
    points: [[f32; 4]; TRAIL_POINTS],
    len: usize,
    /// For the portal, the furthest any pixel can be from where the first
    /// sphere starts, and for the dissolve, how far a sphere grows over the
    /// whole transition, in the units of scene_point in shader.wgsl.
    reach: f32,
    /// How far through its growth a sphere is, from 0 to 1, once it has
    /// grown to each of PACE_POINTS evenly spaced distances out to `reach`,
    /// packed four to a vec4 for the shader. For the portal, it's the
    /// fraction of the screen within that distance of the first sphere's
    /// centre, so the spheres change a similar amount of the picture at
    /// every moment, like the ranks the sweeps use. The dissolve's spheres
    /// grow steadily.
    pace: [[f32; 4]; PACE_POINTS.div_ceil(4)],
    /// When a pixel switches, at each of PACE_POINTS evenly spaced times
    /// from 0 to 1 that the first sphere could reach it, packed like
    /// `pace`. The portal's pace already keeps it steady, so it switches
    /// then. The dissolve's spheres start at different times, so instead
    /// it's when that much of the screen has been reached, which keeps it
    /// steady.
    timing: [[f32; 4]; PACE_POINTS.div_ceil(4)],
}

impl Trail {
    /// Start a portal's trail at `cursor`, over a wallpaper with depths
    /// `grid`, cropped by `crop` to fit a screen `aspect` wide.
    fn new(cursor: [f32; 2], aspect: f32, grid: &DepthGrid, crop: Crop) -> Self {
        let (under, mean) = depths_around(cursor, grid, crop);
        let center = scene_point(cursor, mean, aspect);
        let cells = screen_cells(grid, crop);
        let reach = cells
            .iter()
            .map(|cell| cell.furthest_from(cursor, under, aspect))
            .fold(1e-6, f32::max);

        // How much of the screen is within each distance. For this, each
        // cell counts as its area at its middle and mean depth.
        let mut within = [0.0; PACE_POINTS];
        for cell in &cells {
            let d = distance(cell.middle(aspect), center);
            let step = (d / reach * (PACE_POINTS - 1) as f32).ceil() as usize;
            within[step.min(PACE_POINTS - 1)] += cell.area();
        }
        let mut total = 0.0;
        for w in &mut within {
            total += *w;
            *w = total;
        }
        let mut pace = [[0.0; 4]; PACE_POINTS.div_ceil(4)];
        for (i, w) in within.into_iter().enumerate() {
            pace[i / 4][i % 4] = w / total;
        }

        let mut points = [[0.0; 4]; TRAIL_POINTS];
        points[0] = [cursor[0], cursor[1], 0.0, 0.0];
        Self {
            points,
            len: 1,
            reach,
            pace,
            timing: table(|x| x),
        }
    }

    /// Random points for a dissolve, over a wallpaper with depths `grid`,
    /// cropped by `crop` to fit a screen `aspect` wide. The spheres
    /// start more and more often, each somewhere the ones before haven't
    /// reached yet, and tend to start among the furthest things first.
    /// `seed` picks the points.
    fn scatter(aspect: f32, grid: &DepthGrid, crop: Crop, seed: u32) -> Self {
        let cells = screen_cells(grid, crop);
        // How far away each cell is, as the fraction of the screen nearer
        // than it.
        let mut by_depth: Vec<usize> = (0..cells.len()).collect();
        by_depth.sort_by(|&a, &b| cells[b].depths.mean.total_cmp(&cells[a].depths.mean));
        let total: f32 = cells.iter().map(ScreenCell::area).sum();
        let mut farness = vec![0.0; cells.len()];
        let (mut nearer, mut level, mut depth) = (0.0, 0.0, f32::NAN);
        for i in by_depth {
            // Cells at the same depth are as far as each other.
            if cells[i].depths.mean != depth {
                (nearer, level, depth) = (nearer + level, 0.0, cells[i].depths.mean);
            }
            farness[i] = nearer / total;
            level += cells[i].area();
        }

        // Each sphere grows over this much of the transition, as in
        // portal_arrival in shader.wgsl.
        let growth = 1.0 - 2.0 * PORTAL_EDGE;
        let middles: Vec<[f32; 3]> = cells.iter().map(|cell| cell.middle(aspect)).collect();
        // Where the spheres start if they grow `reach` over the transition,
        // as screen uv and progress, and when the last cell is reached.
        let spheres_for = |reach: f32| {
            let mut random = Random::new(seed);
            let mut spheres = Vec::new();
            let mut reached = vec![f32::MAX; cells.len()];
            for i in 0..DISSOLVE_SPHERES {
                let start = DISSOLVE_SPAWNING
                    * (i as f32 / (DISSOLVE_SPHERES - 1) as f32).powf(DISSOLVE_QUICKENING);
                let weights: Vec<f32> = (cells.iter().zip(&farness).zip(&reached))
                    .map(|((cell, far), &at)| match at <= start {
                        true => 0.0,
                        false => cell.area() * DISSOLVE_FAR.powf(*far),
                    })
                    .collect();
                let Some(cell) = random.pick(&weights).map(|i| &cells[i]) else {
                    break;
                };
                let at = [
                    cell.left + (cell.right - cell.left) * random.next(),
                    cell.top + (cell.bottom - cell.top) * random.next(),
                ];
                let center = scene_point(at, depths_around(at, grid, crop).1, aspect);
                for (r, middle) in reached.iter_mut().zip(&middles) {
                    *r = r.min(start + distance(*middle, center) / reach * growth);
                }
                spheres.push((at, start, center));
            }
            (spheres, reached.into_iter().fold(0.0, f32::max))
        };
        // The slowest the spheres can grow and still reach every cell by
        // the end, so the dissolve takes the whole transition.
        let (mut slow, mut fast) = (0.01, 10.0);
        for _ in 0..20 {
            let reach = f32::sqrt(slow * fast);
            match spheres_for(reach).1 <= growth {
                true => fast = reach,
                false => slow = reach,
            }
        }
        let reach = fast;
        let (mut spheres, _) = spheres_for(reach);
        if spheres.is_empty() {
            let center = scene_point([0.5, 0.5], depths_around([0.5, 0.5], grid, crop).1, aspect);
            spheres.push(([0.5, 0.5], 0.0, center));
        }

        // When the first sphere reaches each cell, as portal_arrival in
        // shader.wgsl works it out, and how much of the screen has been
        // reached by each time. A cell can hold things at quite different
        // depths, so it counts as half its area at its mean depth and a
        // quarter at each end of its range.
        let mut arrivals = Vec::with_capacity(cells.len() * 3);
        for cell in &cells {
            let d = cell.depths;
            for (depth, share) in [(d.least, 0.25), (d.mean, 0.5), (d.greatest, 0.25)] {
                let at = scene_point(cell.center(), depth, aspect);
                let first = (spheres.iter())
                    .map(|&(_, start, center)| {
                        let pace = (distance(at, center) / reach).min(1.0);
                        start + PORTAL_EDGE + pace * growth
                    })
                    .fold(f32::MAX, f32::min);
                arrivals.push((first, cell.area() * share));
            }
        }
        let timing = table(|x| {
            let reached: f32 = (arrivals.iter())
                .filter(|&&(first, _)| first <= x)
                .map(|&(_, area)| area)
                .sum();
            PORTAL_EDGE + reached / total * growth
        });

        let mut points = [[0.0; 4]; TRAIL_POINTS];
        for (point, &(at, start, _)) in points.iter_mut().zip(&spheres) {
            *point = [at[0], at[1], start, 0.0];
        }
        Self {
            points,
            len: spheres.len(),
            reach,
            pace: table(|x| x),
            timing,
        }
    }

    /// Follow the cursor, adding a point once it's moved far enough from
    /// the last. Once the trail is full it stops growing.
    fn follow(&mut self, cursor: [f32; 2], progress: f32, aspect: f32) {
        if self.len == TRAIL_POINTS {
            return;
        }
        let [x, y, ..] = self.points[self.len - 1];
        if ((cursor[0] - x) * aspect).hypot(cursor[1] - y) >= TRAIL_SPACING {
            self.points[self.len] = [cursor[0], cursor[1], progress, 0.0];
            self.len += 1;
        }
    }
}

/// `f` at PACE_POINTS evenly spaced points from 0 to 1, packed four to a
/// vec4 for the shader.
fn table(f: impl Fn(f32) -> f32) -> [[f32; 4]; PACE_POINTS.div_ceil(4)] {
    let mut table = [[0.0; 4]; PACE_POINTS.div_ceil(4)];
    for i in 0..PACE_POINTS {
        table[i / 4][i % 4] = f(i as f32 / (PACE_POINTS - 1) as f32);
    }
    table
}

/// Where a pixel at `screen` is in the scene, as scene_point in
/// shader.wgsl.
fn scene_point(screen: [f32; 2], depth: f32, aspect: f32) -> [f32; 3] {
    [
        (screen[0] - 0.5) * aspect,
        screen[1] - 0.5,
        -(depth + PORTAL_NEAR).ln(),
    ]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>().sqrt()
}

/// The cell of a DepthGrid that image uv `image` is in, on one axis.
fn grid_cell(image: f32) -> usize {
    ((image * GRID_CELLS as f32) as usize).min(GRID_CELLS - 1)
}

/// The range of depths at `screen`, which is somewhere in the range of
/// its cell of `grid` or the ones around it, in case it's just over the
/// edge, and the mean depth of its cell.
fn depths_around(screen: [f32; 2], grid: &DepthGrid, crop: Crop) -> ([f32; 2], f32) {
    let col = grid_cell(crop.to_image(screen[0], 0));
    let row = grid_cell(crop.to_image(screen[1], 1));
    let mut range = [f32::MAX, f32::MIN];
    for r in row.saturating_sub(1)..(row + 2).min(GRID_CELLS) {
        for c in col.saturating_sub(1)..(col + 2).min(GRID_CELLS) {
            let cell = grid.cells[r * GRID_CELLS + c];
            range = [range[0].min(cell.least), range[1].max(cell.greatest)];
        }
    }
    (range, grid.cells[row * GRID_CELLS + col].mean)
}

/// A cell of a wallpaper's DepthGrid, and where it shows on the screen.
struct ScreenCell {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    depths: DepthCell,
}

impl ScreenCell {
    fn area(&self) -> f32 {
        (self.right - self.left) * (self.bottom - self.top)
    }

    /// Its middle on the screen.
    fn center(&self) -> [f32; 2] {
        [
            (self.left + self.right) / 2.0,
            (self.top + self.bottom) / 2.0,
        ]
    }

    /// Where its middle is in the scene, at its mean depth.
    fn middle(&self, aspect: f32) -> [f32; 3] {
        scene_point(self.center(), self.depths.mean, aspect)
    }

    /// The furthest any of its pixels can be in the scene from `screen`, at
    /// a depth in the range `depths`. Its pixels are within its part of the
    /// screen and its range of depths, so the furthest is no further than
    /// one of its corners at one end of that range, from either end of
    /// `depths`.
    fn furthest_from(&self, screen: [f32; 2], depths: [f32; 2], aspect: f32) -> f32 {
        let mut furthest: f32 = 0.0;
        for x in [self.left, self.right] {
            for y in [self.top, self.bottom] {
                for depth in [self.depths.least, self.depths.greatest] {
                    for from in depths {
                        let d = distance(
                            scene_point([x, y], depth, aspect),
                            scene_point(screen, from, aspect),
                        );
                        furthest = furthest.max(d);
                    }
                }
            }
        }
        furthest
    }
}

/// The cells of `grid` that show on a screen, once cropped by `crop`.
fn screen_cells(grid: &DepthGrid, crop: Crop) -> Vec<ScreenCell> {
    let to_screen = |image: f32, axis: usize| crop.to_screen(image, axis);
    let mut cells = Vec::with_capacity(grid.cells.len());
    for row in 0..GRID_CELLS {
        let top = to_screen(row as f32 / GRID_CELLS as f32, 1).max(0.0);
        let bottom = to_screen((row + 1) as f32 / GRID_CELLS as f32, 1).min(1.0);
        for col in 0..GRID_CELLS {
            let left = to_screen(col as f32 / GRID_CELLS as f32, 0).max(0.0);
            let right = to_screen((col + 1) as f32 / GRID_CELLS as f32, 0).min(1.0);
            // Cropped off the screen.
            if top >= bottom || left >= right {
                continue;
            }
            cells.push(ScreenCell {
                left,
                right,
                top,
                bottom,
                depths: grid.cells[row * GRID_CELLS + col],
            });
        }
    }
    cells
}

/// Pseudo-random numbers, the same ones for the same seed.
struct Random(u32);

impl Random {
    fn new(seed: u32) -> Self {
        // Any seed but 0, which would only give 0.
        Self(seed.wrapping_mul(0x9e37_79b9) | 1)
    }

    /// A number from 0 to 1.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1 << 24) as f32
    }

    /// An index into `weights`, more likely the bigger its weight, or None
    /// if they're all 0.
    fn pick(&mut self, weights: &[f32]) -> Option<usize> {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut left = self.next() * total;
        for (i, &w) in weights.iter().enumerate() {
            if left < w {
                return Some(i);
            }
            left -= w;
        }
        weights.iter().rposition(|&w| w > 0.0)
    }
}

/// How far the shader zooms in on each side, at least. Must match MARGIN in
/// shader.wgsl.
const MARGIN: f32 = 0.025;

/// How far the shader zooms in on each side at `intensity`: enough that the
/// biggest shift still pulls in real pixels from beyond the screen, and at
/// least MARGIN. As crop in shader.wgsl.
fn margin(intensity: f32) -> f32 {
    MARGIN.max(0.5 * intensity)
}

/// Which part of a wallpaper shows: zoomed in `zoom` times, where 1 shows
/// all of it that fits the screen's shape, around `center`, in image uv
/// from the top left. The centre moves in as far as it has to for the
/// image to cover the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub center: [f32; 2],
    pub zoom: f32,
}

impl Default for View {
    /// All of the image that fits the screen's shape, from its middle.
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            zoom: 1.0,
        }
    }
}

/// The part of an image that shows on a screen: `scale` of it on each
/// axis, around `center`, in image uv. As crop in shader.wgsl.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Crop {
    scale: [f32; 2],
    center: [f32; 2],
}

impl Crop {
    /// How `view` shows an image of size `image` on a screen of size
    /// `screen`. With `pan`, from 0 to 1, the horizontal centre goes from
    /// as far left as the image allows to as far right instead.
    fn new(image: (u32, u32), screen: (u32, u32), view: View, pan: Option<f32>) -> Self {
        let cover = cover_uv_scale(image, screen);
        let zoom = view.zoom.max(1.0);
        let scale = [cover[0] / zoom, cover[1] / zoom];
        let mut center = view.center;
        if let Some(pan) = pan {
            center[0] = scale[0] / 2.0 + pan.clamp(0.0, 1.0) * (1.0 - scale[0]);
        }
        let fit = |axis: usize| center[axis].clamp(scale[axis] / 2.0, 1.0 - scale[axis] / 2.0);
        Self {
            scale,
            center: [fit(0), fit(1)],
        }
    }

    /// Zoomed in by the margin at `intensity`, as crop in shader.wgsl.
    fn zoomed(self, intensity: f32) -> Self {
        let zoom = 1.0 - 2.0 * margin(intensity);
        Self {
            scale: [self.scale[0] * zoom, self.scale[1] * zoom],
            center: self.center,
        }
    }

    /// Screen uv to image uv, on `axis`.
    fn to_image(self, screen: f32, axis: usize) -> f32 {
        self.center[axis] + (screen - 0.5) * self.scale[axis]
    }

    /// Image uv to screen uv, on `axis`.
    fn to_screen(self, image: f32, axis: usize) -> f32 {
        0.5 + (image - self.center[axis]) / self.scale[axis]
    }
}

/// A wallpaper's images decoded into memory, ready to upload. Decoding is
/// the slow part, so it can happen off the main thread.
pub struct DecodedWallpaper {
    color: image::RgbaImage,
    /// None if the depth map couldn't be loaded, so the wallpaper still
    /// shows, just without parallax.
    depth: Option<DepthMap>,
    /// From `DepthMap::ranks` and `DepthMap::height_ranks`, for ordering
    /// transitions.
    ranks: Vec<[f32; 2]>,
    /// From `DepthMap::grid`, for the portal.
    grid: DepthGrid,
    /// From `DepthMap::ground`, for the tide.
    ground: Ground,
    files: WallpaperFiles,
}

/// The color and depth images a wallpaper is loaded from.
pub type WallpaperFiles = (PathBuf, PathBuf);

impl DecodedWallpaper {
    /// Load a baked wallpaper. If it's bigger than any of `screens` needs,
    /// it's scaled down, which makes every frame cheaper to draw and
    /// saves memory. To keep the detail for a `View` that zooms in, pass
    /// the screen's size times the zoom.
    pub fn load(color_path: &Path, depth_path: &Path, screens: &[(u32, u32)]) -> Result<Self> {
        let mut color = image::open(color_path)
            .with_context(|| format!("failed to load image: {}", color_path.display()))?
            .to_rgba8();
        let mut depth = match load_depth_map(depth_path) {
            Ok(map) => Some(map),
            Err(e) => {
                warn!("using a flat depth map: {e:#}");
                None
            }
        };
        if let Some(scale) = fit_scale(color.dimensions(), screens) {
            let (w, h) = scale_size(color.dimensions(), scale);
            debug!(w, h, "scaling wallpaper down to fit the screens");
            color = imageops::resize(&color, w, h, FilterType::Triangle);
            depth = depth.map(|map| {
                let (w, h) = scale_size((map.width, map.height), scale);
                map.resized(w, h)
            });
        }
        let ground = depth.as_ref().map_or(Ground::LEVEL, DepthMap::ground);
        // A flat wallpaper is all one depth, so it changes all at once,
        // halfway through a transition.
        let ranks = match &depth {
            Some(map) => (map.ranks().into_iter())
                .zip(map.height_ranks(&ground))
                .map(|(depth, height)| [depth, height])
                .collect(),
            None => vec![[0.5, 0.5]; RANK_POINTS],
        };
        let grid = depth.as_ref().map_or_else(DepthGrid::flat, DepthMap::grid);
        Ok(Self {
            color,
            depth,
            ranks,
            grid,
            ground,
            files: (color_path.to_path_buf(), depth_path.to_path_buf()),
        })
    }

    /// A copy shrunk so neither side is over `max` pixels, or None if it's
    /// that small already.
    fn within(&self, max: u32) -> Option<Self> {
        let scale = limit_scale(self.color.dimensions(), max)?;
        let fit = |size| {
            let (w, h) = scale_size(size, scale);
            (w.min(max), h.min(max))
        };
        let (w, h) = fit(self.color.dimensions());
        warn!(w, h, "shrinking a wallpaper to the most the GPU takes");
        Some(Self {
            color: imageops::resize(&self.color, w, h, FilterType::Triangle),
            depth: self.depth.as_ref().map(|map| {
                let (w, h) = fit((map.width, map.height));
                map.resized(w, h)
            }),
            ranks: self.ranks.clone(),
            grid: self.grid.clone(),
            ground: self.ground,
            files: self.files.clone(),
        })
    }
}

/// How much to shrink an image so it still covers each of `screens` once
/// cropped to their shape, or None if it isn't bigger than they need.
fn fit_scale((width, height): (u32, u32), screens: &[(u32, u32)]) -> Option<f32> {
    let needed = screens
        .iter()
        .map(|&(w, h)| f32::max(w as f32 / width as f32, h as f32 / height as f32))
        .fold(0.0, f32::max)
        // The shader zooms in a little, which needs a little more detail.
        // A stronger parallax zooms in a few percent more, too little to see.
        / (1.0 - 2.0 * MARGIN);
    (needed > 0.0 && needed < 1.0).then_some(needed)
}

/// How much to shrink an image so neither side is over `max` pixels, or
/// None if it isn't.
fn limit_scale((width, height): (u32, u32), max: u32) -> Option<f32> {
    let longest = width.max(height);
    (longest > max).then(|| max as f32 / longest as f32)
}

/// `size` multiplied by `scale`, rounded up.
fn scale_size((width, height): (u32, u32), scale: f32) -> (u32, u32) {
    let scale = |n: u32| ((n as f32 * scale).ceil() as u32).max(1);
    (scale(width), scale(height))
}

/// A wallpaper's textures on the GPU. Cloning shares the textures, so
/// outputs showing the same wallpaper upload it once.
#[derive(Clone)]
pub struct Wallpaper {
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    ranks: wgpu::TextureView,
    /// Where things are in the scene, roughly, for the portal.
    grid: DepthGrid,
    /// Where the ground is, for the tide.
    ground: Ground,
    /// Image size in pixels, for cropping to the screen's aspect ratio.
    size: (u32, u32),
    pub files: WallpaperFiles,
    /// Which part of it shows. Set it before showing the wallpaper, or with
    /// `OutputRenderState::set_view` once it's showing.
    pub view: View,
}

/// A short effect played over the wallpaper showing, such as as a phone
/// unlocks. Each one moves through the depth of the scene for a moment and
/// leaves it exactly as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    /// A ripple spreads from a point like a drop in water, over the
    /// surface there first and on through the scene, wrapping round things
    /// nearer and further away, as the portal does.
    Ripple,
    /// A swell rolls from the nearest things to the furthest, each pushing
    /// toward the viewer for a moment as it passes.
    Swell,
    /// A band of light passes through the scene from near to far, and each
    /// surface lifts a little as it catches it.
    Glint,
}

impl Effect {
    pub const ALL: [Effect; 3] = [Effect::Ripple, Effect::Swell, Effect::Glint];

    /// Its name, as front ends can ask for it.
    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Ripple => "ripple",
            Effect::Swell => "swell",
            Effect::Glint => "glint",
        }
    }
}

/// An effect that's under way.
#[derive(Clone, Copy)]
struct ActiveEffect {
    effect: Effect,
    start: Instant,
    duration: Duration,
    origin: [f32; 2],
}

/// A transition to another wallpaper that's under way.
struct ActiveTransition {
    next: Wallpaper,
    start: Instant,
    duration: Duration,
    style: Transition,
    trail: Trail,
    /// Frames drawn so far, logged at the end to show how smooth it was.
    frames: u32,
}

pub struct OutputRenderState {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    bind_group: wgpu::BindGroup,
    current: Wallpaper,
    transition: Option<ActiveTransition>,
    uniform_buffer: wgpu::Buffer,
    current_offset: (f32, f32),
    pub target_offset: (f32, f32),
    /// The intensity `step_and_write` last drew with, which sets the margin.
    intensity: f32,
    /// From `set_pan`.
    pan: Option<f32>,
    /// Whether the view or pan changed since `step_and_write` last ran.
    view_changed: bool,
    /// From `start_effect`, until it's run its course.
    effect: Option<ActiveEffect>,
}

impl OutputRenderState {
    pub fn new(
        renderer: &Renderer,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        wallpaper: Wallpaper,
    ) -> Self {
        let uniform_buffer = renderer.create_uniform_buffer();
        let bind_group = renderer.create_bind_group(&wallpaper, &wallpaper, &uniform_buffer);
        Self {
            surface,
            config,
            bind_group,
            current: wallpaper,
            transition: None,
            uniform_buffer,
            current_offset: (0.0, 0.0),
            target_offset: (0.0, 0.0),
            intensity: 0.0,
            pan: None,
            view_changed: false,
            effect: None,
        }
    }

    /// The wallpaper this output is showing, or changing to if a transition
    /// is under way.
    pub fn showing(&self) -> &Wallpaper {
        match &self.transition {
            Some(t) => &t.next,
            None => &self.current,
        }
    }

    /// Show another part of the wallpaper: the one showing, or the one a
    /// transition is changing to.
    pub fn set_view(&mut self, view: View) {
        match &mut self.transition {
            Some(t) => t.next.view = view,
            None => self.current.view = view,
        }
        self.view_changed = true;
    }

    /// Pan across the wallpaper, from as far left as it goes at 0 to as far
    /// right at 1, in place of its view's horizontal centre, or stop with
    /// None. For following the pages of a desktop or launcher.
    pub fn set_pan(&mut self, pan: Option<f32>) {
        if pan != self.pan {
            self.pan = pan;
            self.view_changed = true;
        }
    }

    /// Show a wallpaper straight away, cancelling any transition.
    pub fn set_wallpaper(&mut self, renderer: &Renderer, wallpaper: Wallpaper) {
        self.transition = None;
        self.bind_group = renderer.create_bind_group(&wallpaper, &wallpaper, &self.uniform_buffer);
        self.current = wallpaper;
    }

    /// Start a transition to `next`. A transition that is already running
    /// skips to its end first.
    pub fn start_transition(
        &mut self,
        renderer: &Renderer,
        next: Wallpaper,
        duration: Duration,
        style: Transition,
    ) {
        if let Some(running) = self.transition.take() {
            self.current = running.next;
        }
        self.bind_group = renderer.create_bind_group(&self.current, &next, &self.uniform_buffer);
        let (x, y) = self.current_offset;
        let crop = self.crop(&self.current).zoomed(self.intensity);
        let trail = match style {
            Transition::Dissolve => {
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |t| t.subsec_nanos());
                Trail::scatter(self.aspect(), &self.current.grid, crop, seed)
            }
            _ => Trail::new([x + 0.5, y + 0.5], self.aspect(), &self.current.grid, crop),
        };
        self.transition = Some(ActiveTransition {
            next,
            start: Instant::now(),
            duration,
            style,
            trail,
            frames: 0,
        });
    }

    /// Play `effect` over the wallpaper showing for `duration`, starting
    /// from `origin` in screen uv, from 0 to 1 across and down, where it
    /// starts from one place. An effect already playing stops. Effects show
    /// over a wallpaper at rest, not during a transition.
    pub fn start_effect(&mut self, effect: Effect, duration: Duration, origin: [f32; 2]) {
        self.effect = Some(ActiveEffect {
            effect,
            start: Instant::now(),
            duration,
            origin,
        });
    }

    /// Whether an effect is playing.
    pub fn is_playing_effect(&self) -> bool {
        self.effect.is_some()
    }

    /// Once a transition has run its course, make its wallpaper the current
    /// one. Call after drawing, so the final frame of the transition has
    /// been shown.
    pub fn finish_transition_if_done(&mut self, renderer: &Renderer) {
        let done = self
            .transition
            .as_ref()
            .is_some_and(|t| t.start.elapsed() >= t.duration);
        if done && let Some(finished) = self.transition.take() {
            debug!(frames = finished.frames, "transition finished");
            self.set_wallpaper(renderer, finished.next);
        }
    }

    fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    /// The part of `wallpaper` that shows, before the margin.
    fn crop(&self, wallpaper: &Wallpaper) -> Crop {
        let screen = (self.config.width, self.config.height);
        Crop::new(wallpaper.size, screen, wallpaper.view, self.pan)
    }

    /// Step the lerp 30% of the way toward target_offset and write the new
    /// uniforms. Returns whether anything changed, so callers can skip
    /// redrawing once the offset has settled and no transition is running.
    pub fn step_and_write(&mut self, queue: &wgpu::Queue, intensity: f32) -> bool {
        self.intensity = intensity;
        let (tx, ty) = self.target_offset;
        let (cx, cy) = self.current_offset;
        let nx = cx + (tx - cx) * 0.3;
        let ny = cy + (ty - cy) * 0.3;
        self.current_offset = (nx, ny);

        let aspect = self.aspect();
        let crop = self.crop(&self.current);
        let (progress, next, style, trail) = match &mut self.transition {
            Some(t) => {
                t.frames += 1;
                let progress = transition_progress(t.start.elapsed(), t.duration);
                if t.style == Transition::Portal {
                    t.trail.follow([nx + 0.5, ny + 0.5], progress, aspect);
                }
                (progress, &t.next, t.style, Some(t.trail))
            }
            None => (0.0, &self.current, Transition::default(), None),
        };
        let next_crop = Crop::new(
            next.size,
            (self.config.width, self.config.height),
            next.view,
            self.pan,
        );
        // An effect that's run its course draws one last frame without it.
        let effect = self.effect.filter(|e| e.start.elapsed() < e.duration);
        let had_effect = self.effect.is_some();
        self.effect = effect;
        let uniforms = Uniforms {
            cursor_offset: [nx, ny],
            intensity,
            progress,
            uv_scale: crop.scale,
            next_uv_scale: next_crop.scale,
            uv_center: crop.center,
            next_uv_center: next_crop.center,
            style: shader_style(style),
            aspect,
            trail_len: trail.map_or(0, |t| t.len as u32),
            portal_reach: trail.map_or(0.0, |t| t.reach),
            trail: trail.map_or([[0.0; 4]; TRAIL_POINTS], |t| t.points),
            portal_pace: trail.map_or(Zeroable::zeroed(), |t| t.pace),
            portal_timing: trail.map_or(Zeroable::zeroed(), |t| t.timing),
            ground: self.current.ground.0,
            next_ground: next.ground.0,
            effect: effect.map_or(0, |e| shader_effect(e.effect)),
            effect_progress: effect.map_or(0.0, |e| effect_progress(e.start.elapsed(), e.duration)),
            effect_origin: effect.map_or([0.5, 0.5], |e| e.origin),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let moved = (nx - cx).abs() > 1e-5 || (ny - cy).abs() > 1e-5;
        let view_changed = std::mem::take(&mut self.view_changed);
        moved || view_changed || self.transition.is_some() || had_effect
    }
}

/// The shader's number for each transition. Must match the constants in
/// shader.wgsl.
fn shader_style(style: Transition) -> u32 {
    match style {
        Transition::SweepIn => 0,
        Transition::SweepOut => 1,
        Transition::Morph => 2,
        Transition::Dissolve => 3,
        Transition::Portal => 4,
        Transition::TideIn => 5,
        Transition::TideOut => 6,
    }
}

/// The shader's number for each effect, from 1, as 0 is none. Must match
/// the constants in shader.wgsl.
fn shader_effect(effect: Effect) -> u32 {
    match effect {
        Effect::Ripple => 1,
        Effect::Swell => 2,
        Effect::Glint => 3,
    }
}

/// How far through an effect we are, from 0 to 1, evenly: each effect
/// eases itself in the shader.
fn effect_progress(elapsed: Duration, duration: Duration) -> f32 {
    if duration.is_zero() {
        return 1.0;
    }
    (elapsed.as_secs_f32() / duration.as_secs_f32()).min(1.0)
}

/// How far through a transition we are, eased in and out so it
/// starts and settles gently. Reaches exactly 1 once `duration` has passed.
fn transition_progress(elapsed: Duration, duration: Duration) -> f32 {
    if duration.is_zero() {
        return 1.0;
    }
    let t = (elapsed.as_secs_f32() / duration.as_secs_f32()).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of the image to show on each axis so it fills a screen of a
/// different aspect ratio without stretching. The overflowing axis gets a
/// value below 1 and is cropped equally on both sides.
fn cover_uv_scale(image: (u32, u32), screen: (u32, u32)) -> [f32; 2] {
    let image_aspect = image.0 as f32 / image.1 as f32;
    let screen_aspect = screen.0 as f32 / screen.1 as f32;
    if image_aspect > screen_aspect {
        [screen_aspect / image_aspect, 1.0]
    } else {
        [1.0, image_aspect / screen_aspect]
    }
}

pub struct Renderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The format it draws in, which surfaces must be configured with.
    pub format: wgpu::TextureFormat,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    _depth_placeholder: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl Renderer {
    /// Draw with `adapter`, from `instance`, to surfaces in `format`. The
    /// caller picks all three, so each front end can choose the backends
    /// and format its platform offers.
    pub async fn new(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        info!(name = adapter.get_info().name, "using GPU");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("shiftpaper"),
                required_features: wgpu::Features::empty(),
                // What even GLES phones manage, with the texture sizes this
                // adapter supports.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .context("failed to create device")?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shiftpaper_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The wallpaper being transitioned to: color, then depth.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth ranks for ordering transitions: current, then next.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shiftpaper_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shiftpaper_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // A screen without an sRGB format gets colour the shader encodes itself.
        let encode_srgb = if needs_srgb_encoding(format) {
            1.0
        } else {
            0.0
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shiftpaper_rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("ENCODE_SRGB", encode_srgb)],
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shiftpaper_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let depth_placeholder = create_placeholder_depth(&device, &queue);
        let depth_view = depth_placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            format,
            pipeline,
            bind_group_layout,
            sampler,
            _depth_placeholder: depth_placeholder,
            depth_view,
        })
    }

    /// Create a uniform buffer for an output. Called once per output when
    /// its surface is set up, and filled in by `step_and_write` before
    /// the first draw.
    pub fn create_uniform_buffer(&self) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&Uniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Load a baked wallpaper from disk onto the GPU, scaled down to fit
    /// `screens` if it's bigger than they need.
    pub fn load_wallpaper(
        &self,
        color: &Path,
        depth: &Path,
        screens: &[(u32, u32)],
    ) -> Result<Wallpaper> {
        Ok(self.upload_wallpaper(&DecodedWallpaper::load(color, depth, screens)?))
    }

    pub fn upload_wallpaper(&self, decoded: &DecodedWallpaper) -> Wallpaper {
        // A very wide or tall photo can be bigger than the GPU takes, even
        // once shrunk to the screen.
        let shrunk = decoded.within(self.device.limits().max_texture_dimension_2d);
        let decoded = shrunk.as_ref().unwrap_or(decoded);
        Wallpaper {
            color: self.upload_color(&decoded.color),
            depth: match &decoded.depth {
                Some(map) => self.upload_depth_map(map),
                None => self.depth_view.clone(),
            },
            ranks: self.upload_ranks(&decoded.ranks),
            grid: decoded.grid.clone(),
            ground: decoded.ground,
            size: decoded.color.dimensions(),
            files: decoded.files.clone(),
            view: View::default(),
        }
    }

    fn upload_ranks(&self, ranks: &[[f32; 2]]) -> wgpu::TextureView {
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("depth_ranks"),
                size: wgpu::Extent3d {
                    width: ranks.len() as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D1,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(ranks),
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn upload_depth_map(&self, depth: &DepthMap) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: depth.width,
            height: depth.height,
            depth_or_array_layers: 1,
        };

        // R16Unorm would hold the map as it is, but it's optional on phones.
        // A 16-bit float loses precision only near 1, where it's a fraction
        // of a pixel at any sensible intensity.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&depth.to_f16()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * depth.width),
                rows_per_image: Some(depth.height),
            },
            size,
        );

        debug!(
            w = depth.width,
            h = depth.height,
            "uploaded depth map to GPU"
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Bind the current wallpaper and the one being transitioned to, which
    /// is the same wallpaper when there's no transition.
    fn create_bind_group(
        &self,
        current: &Wallpaper,
        next: &Wallpaper,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shiftpaper_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&current.color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&current.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&next.color),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&next.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&current.ranks),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&next.ranks),
                },
            ],
        })
    }

    fn upload_color(&self, img: &image::RgbaImage) -> wgpu::TextureView {
        let (w, h) = img.dimensions();

        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wallpaper_color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            size,
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Draw and present one frame. Returns false if nothing was presented,
    /// so the caller can retry on the next frame callback.
    pub fn render_frame(&self, output: &OutputRenderState) -> bool {
        let frame = match output.surface.get_current_texture() {
            // A suboptimal frame is still usable, and the front end
            // reconfigures the surface whenever its size changes anyway.
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return false,
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                output.surface.configure(&self.device, &output.config);
                return false;
            }
            CurrentSurfaceTexture::Validation => {
                tracing::warn!("surface validation error while acquiring a frame");
                return false;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shiftpaper_enc"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shiftpaper_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &output.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        true
    }
}

fn create_placeholder_depth(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_placeholder"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Zero as a 16-bit float is all zero bytes, like the data below.
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 2],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2),
            rows_per_image: Some(1),
        },
        size,
    );
    tex
}

/// Whether `format` stores colour sRGB-encoded without the GPU encoding it
/// on write, so the shader has to: an 8 or 10 bit unorm format without the
/// sRGB suffix. Float formats hold linear colour, so they're left alone.
fn needs_srgb_encoding(format: wgpu::TextureFormat) -> bool {
    use wgpu::TextureFormat::*;
    matches!(format, Rgba8Unorm | Bgra8Unorm | Rgb10a2Unorm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_unorm_formats_without_srgb_are_encoded_by_the_shader() {
        use wgpu::TextureFormat::*;
        for format in [Rgba8Unorm, Bgra8Unorm, Rgb10a2Unorm] {
            assert!(needs_srgb_encoding(format), "{format:?}");
        }
        for format in [Rgba8UnormSrgb, Bgra8UnormSrgb, Rgba16Float] {
            assert!(!needs_srgb_encoding(format), "{format:?}");
        }
    }

    fn assert_close(actual: [f32; 2], expected: [f32; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < 1e-6 && (actual[1] - expected[1]).abs() < 1e-6,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn matching_aspect_is_not_cropped() {
        assert_close(cover_uv_scale((3840, 2160), (2560, 1440)), [1.0, 1.0]);
    }

    #[test]
    fn wider_image_is_cropped_horizontally() {
        // A 16:9 image on a 16:10 laptop panel shows 90% of its width.
        assert_close(cover_uv_scale((3840, 2160), (2560, 1600)), [0.9, 1.0]);
    }

    #[test]
    fn taller_image_is_cropped_vertically() {
        assert_close(cover_uv_scale((1000, 2000), (2000, 1000)), [1.0, 0.25]);
    }

    #[test]
    fn a_view_zooms_in_around_its_centre() {
        let view = View {
            center: [0.3, 0.6],
            zoom: 2.0,
        };
        let crop = Crop::new((3840, 2160), (2560, 1440), view, None);
        assert_close(crop.scale, [0.5, 0.5]);
        assert_close(crop.center, [0.3, 0.6]);
        assert_close([crop.to_image(0.5, 0), crop.to_image(0.0, 1)], [0.3, 0.35]);
        assert_close(
            [crop.to_screen(0.3, 0), crop.to_screen(0.35, 1)],
            [0.5, 0.0],
        );
    }

    #[test]
    fn a_view_keeps_the_screen_covered() {
        let corner = View {
            center: [0.0, 1.0],
            zoom: 2.0,
        };
        let crop = Crop::new((3840, 2160), (2560, 1440), corner, None);
        assert_close(crop.center, [0.25, 0.75]);
        // It can't zoom out past the whole image.
        let wide = View {
            center: [0.2, 0.2],
            zoom: 0.5,
        };
        let crop = Crop::new((3840, 2160), (2560, 1440), wide, None);
        assert_close(crop.scale, [1.0, 1.0]);
        assert_close(crop.center, [0.5, 0.5]);
    }

    #[test]
    fn panning_crosses_the_whole_image() {
        // An image three screens wide.
        let view = View::default();
        let at = |pan| Crop::new((3000, 1000), (1000, 1000), view, Some(pan));
        assert_close(at(0.0).center, [1.0 / 6.0, 0.5]);
        assert_close(at(0.5).center, [0.5, 0.5]);
        assert_close(at(1.0).center, [5.0 / 6.0, 0.5]);
        assert_close(
            [at(0.0).to_image(0.0, 0), at(1.0).to_image(1.0, 0)],
            [0.0, 1.0],
        );
        // Zoomed in, there's further to go.
        let zoomed = View { zoom: 2.0, ..view };
        let crop = Crop::new((3000, 1000), (1000, 1000), zoomed, Some(1.0));
        assert_close(crop.center, [11.0 / 12.0, 0.5]);
    }

    #[test]
    fn transition_numbers_match_the_shader() {
        let src = include_str!("shader.wgsl");
        for (name, style) in [
            ("SWEEP_IN", Transition::SweepIn),
            ("SWEEP_OUT", Transition::SweepOut),
            ("MORPH", Transition::Morph),
            ("DISSOLVE", Transition::Dissolve),
            ("PORTAL", Transition::Portal),
            ("TIDE_IN", Transition::TideIn),
            ("TIDE_OUT", Transition::TideOut),
        ] {
            let line = format!("const {name}: u32 = {}u;", shader_style(style));
            assert!(src.contains(&line), "{line}");
        }
    }

    #[test]
    fn effect_numbers_match_the_shader() {
        let src = include_str!("shader.wgsl");
        assert!(src.contains("const NO_EFFECT: u32 = 0u;"));
        for (name, effect) in [
            ("EFFECT_RIPPLE", Effect::Ripple),
            ("EFFECT_SWELL", Effect::Swell),
            ("EFFECT_GLINT", Effect::Glint),
        ] {
            let line = format!("const {name}: u32 = {}u;", shader_effect(effect));
            assert!(src.contains(&line), "{line}");
        }
    }

    #[test]
    fn effects_have_their_own_names() {
        for effect in Effect::ALL {
            let same = Effect::ALL.iter().filter(|e| e.as_str() == effect.as_str());
            assert_eq!(same.count(), 1, "{}", effect.as_str());
        }
    }

    #[test]
    fn effects_run_evenly_to_their_end() {
        let second = Duration::from_secs(1);
        assert_eq!(effect_progress(Duration::ZERO, second), 0.0);
        assert_eq!(effect_progress(second / 4, second), 0.25);
        assert_eq!(effect_progress(second * 2, second), 1.0);
        assert_eq!(effect_progress(second, Duration::ZERO), 1.0);
    }

    #[test]
    fn portal_constants_match_the_shader() {
        let src = include_str!("shader.wgsl");
        for line in [
            format!("const PORTAL_NEAR: f32 = {PORTAL_NEAR:?};"),
            format!("const PORTAL_EDGE: f32 = {PORTAL_EDGE:?};"),
            format!("const PACE_POINTS: u32 = {PACE_POINTS}u;"),
            format!("const TRAIL_POINTS: u32 = {TRAIL_POINTS}u;"),
        ] {
            assert!(src.contains(&line), "{line}");
        }
    }

    /// How far a pixel at `screen` is from where a trail from `cursor`
    /// starts, as portal_arrival in shader.wgsl works it out.
    fn portal_distance(
        map: &DepthMap,
        crop: Crop,
        aspect: f32,
        cursor: [f32; 2],
        screen: [f32; 2],
    ) -> f32 {
        // As crop, load_depth and scene_point in shader.wgsl.
        let point = |s: [f32; 2]| {
            let image = |axis: usize| crop.to_image(s[axis], axis);
            let x = ((image(0) * map.width as f32) as u32).min(map.width - 1);
            let y = ((image(1) * map.height as f32) as u32).min(map.height - 1);
            let depth = f32::from(map.data[(y * map.width + x) as usize]) / 65535.0;
            [
                (s[0] - 0.5) * aspect,
                s[1] - 0.5,
                -(depth + PORTAL_NEAR).ln(),
            ]
        };
        let (p, c) = (point(screen), point(cursor));
        (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f32>().sqrt()
    }

    /// As pace in shader.wgsl.
    fn pace(trail: &Trail, distance: f32) -> f32 {
        let x = (distance / trail.reach).clamp(0.0, 1.0) * (PACE_POINTS - 1) as f32;
        let i = (x as usize).min(PACE_POINTS - 2);
        let (a, b) = (
            trail.pace[i / 4][i % 4],
            trail.pace[(i + 1) / 4][(i + 1) % 4],
        );
        a + (b - a) * (x - i as f32)
    }

    /// A depth map wider than the screen, with depths from `depth` at
    /// each pixel.
    fn depth_map(depth: impl Fn(u32, u32) -> f32) -> DepthMap {
        let (width, height) = (300, 160);
        let data = (0..width * height)
            .map(|i| (depth(i % width, i / width) * 65535.0) as u16)
            .collect();
        DepthMap {
            data,
            width,
            height,
        }
    }

    /// Pixel centres all over a 160 x 100 screen.
    fn screen_pixels() -> impl Iterator<Item = [f32; 2]> {
        (0..160 * 100).map(|i| {
            [
                ((i % 160) as f32 + 0.5) / 160.0,
                ((i / 160) as f32 + 0.5) / 100.0,
            ]
        })
    }

    const CURSORS: [[f32; 2]; 4] = [[0.5, 0.5], [0.0, 0.0], [0.9, 0.2], [0.3, 0.97]];

    /// How `depth_map`s show on a 1600 x 1000 screen.
    fn cover(map: &DepthMap) -> Crop {
        Crop::new((map.width, map.height), (1600, 1000), View::default(), None)
    }

    #[test]
    fn the_portal_reaches_every_pixel() {
        // A slope with sharp bumps all over it.
        let map =
            depth_map(|x, y| y as f32 / 160.0 * 0.6 + ((x * 7 + y * 13) % 23) as f32 / 22.0 * 0.4);
        // Zoomed in off centre, too.
        let view = View {
            center: [0.2, 0.7],
            zoom: 1.7,
        };
        let zoomed = Crop::new((map.width, map.height), (1600, 1000), view, None);
        for crop in [cover(&map), zoomed] {
            for cursor in CURSORS {
                let trail = Trail::new(cursor, 1.6, &map.grid(), crop);
                for screen in screen_pixels() {
                    let distance = portal_distance(&map, crop, 1.6, cursor, screen);
                    // To within a pixel.
                    assert!(
                        distance <= trail.reach + 1e-3,
                        "{crop:?} {cursor:?} {screen:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_portal_changes_the_picture_steadily() {
        // A slope, and a slope with a few specks far away, like sky through
        // trees, which shouldn't hold the rest up.
        let slope = |_, y| y as f32 / 160.0;
        let specks = |x, y| {
            if (x + y * 3) % 97 == 0 {
                0.0
            } else {
                0.5 + y as f32 / 320.0
            }
        };
        for map in [depth_map(slope), depth_map(specks)] {
            let crop = cover(&map);
            for cursor in CURSORS {
                let trail = Trail::new(cursor, 1.6, &map.grid(), crop);
                let paces: Vec<f32> = screen_pixels()
                    .map(|s| pace(&trail, portal_distance(&map, crop, 1.6, cursor, s)))
                    .collect();
                for part in [0.25, 0.5, 0.75] {
                    let done =
                        paces.iter().filter(|&&p| p <= part).count() as f32 / paces.len() as f32;
                    assert!((done - part).abs() < 0.1, "{cursor:?} {part} {done}");
                }
            }
        }
    }

    /// When the pixel at `screen` switches in a dissolve, as portal_arrival
    /// in shader.wgsl works it out, but without the noise.
    fn dissolve_switches(trail: &Trail, map: &DepthMap, crop: Crop, screen: [f32; 2]) -> f32 {
        let growth = 1.0 - 2.0 * PORTAL_EDGE;
        let first = (trail.points[..trail.len].iter())
            .map(|&[x, y, start, _]| {
                let d = portal_distance(map, crop, 1.6, [x, y], screen);
                start + PORTAL_EDGE + pace(trail, d) * growth
            })
            .fold(2.0, f32::min);
        // As timing in shader.wgsl.
        let x = first.clamp(0.0, 1.0) * (PACE_POINTS - 1) as f32;
        let i = (x as usize).min(PACE_POINTS - 2);
        let (a, b) = (
            trail.timing[i / 4][i % 4],
            trail.timing[(i + 1) / 4][(i + 1) % 4],
        );
        a + (b - a) * (x - i as f32)
    }

    #[test]
    fn the_dissolve_changes_the_picture_steadily() {
        // A slope with sharp bumps all over it.
        let map =
            depth_map(|x, y| y as f32 / 160.0 * 0.6 + ((x * 7 + y * 13) % 23) as f32 / 22.0 * 0.4);
        let crop = cover(&map);
        for seed in 0..4 {
            let trail = Trail::scatter(1.6, &map.grid(), crop, seed);
            let switches: Vec<f32> = screen_pixels()
                .map(|s| dissolve_switches(&trail, &map, crop, s))
                .collect();
            for part in [0.25, 0.5, 0.75, 1.0] {
                let done =
                    switches.iter().filter(|&&p| p <= part).count() as f32 / switches.len() as f32;
                assert!((done - part).abs() < 0.1, "{seed} {part} {done}");
            }
        }
    }

    #[test]
    fn the_dissolve_starts_far_away_and_quickens() {
        // Far away at the top.
        let map = depth_map(|_, y| y as f32 / 160.0);
        let crop = cover(&map);
        let mut first_heights = 0.0;
        for seed in 0..20 {
            let trail = Trail::scatter(1.6, &map.grid(), crop, seed);
            first_heights += trail.points[0][1] / 20.0;
            let starts: Vec<f32> = trail.points[..trail.len].iter().map(|p| p[2]).collect();
            assert!(trail.len > 1 && starts[0] == 0.0, "{starts:?}");
            for gaps in starts.windows(3) {
                assert!(gaps[2] - gaps[1] <= gaps[1] - gaps[0] + 1e-6, "{starts:?}");
            }
        }
        assert!(first_heights < 0.4, "{first_heights}");
    }

    #[test]
    fn the_trail_follows_the_cursor_in_steps() {
        let whole = Crop::new((1, 1), (1, 1), View::default(), None);
        let mut trail = Trail::new([0.5, 0.5], 1.0, &DepthGrid::flat(), whole);
        trail.follow([0.51, 0.5], 0.1, 1.0);
        assert_eq!(trail.len, 1, "a small move doesn't add a point");
        trail.follow([0.6, 0.5], 0.2, 1.0);
        assert_eq!(trail.len, 2);
        assert_eq!(trail.points[1], [0.6, 0.5, 0.2, 0.0]);
        for i in 0..100 {
            trail.follow([0.6 + 0.1 * i as f32, 0.5], 0.3, 1.0);
        }
        assert_eq!(trail.len, TRAIL_POINTS, "it stops when full");
    }

    #[test]
    fn tide_near_matches_the_shader() {
        let line = format!("const TIDE_NEAR: f32 = {:?};", crate::depth::TIDE_NEAR);
        assert!(include_str!("shader.wgsl").contains(&line), "{line}");
    }

    #[test]
    fn margin_matches_the_shader() {
        let line = format!("const MARGIN: f32 = {MARGIN};");
        assert!(include_str!("shader.wgsl").contains(&line), "{line}");
    }

    #[test]
    fn the_margin_covers_the_biggest_shift() {
        for intensity in [0.0, 0.025, 0.05, 0.08, 0.2] {
            // Offsets reach 0.5 and depths 1, so nothing shifts further.
            let shift = 0.5 * intensity;
            assert!(margin(intensity) >= shift.max(MARGIN), "{intensity}");
        }
        assert!(include_str!("shader.wgsl").contains("max(MARGIN, 0.5 * u.intensity)"));
    }

    #[test]
    fn large_images_shrink_to_cover_the_largest_screen() {
        // A 16:10 screen is wider than a 3:2 image, so width sets the size.
        let scale = fit_scale((6000, 4000), &[(1920, 1080), (2560, 1600)]).unwrap();
        assert!((scale - 2560.0 / 6000.0 / 0.95).abs() < 1e-6, "{scale}");
    }

    #[test]
    fn images_shrink_to_what_the_gpu_takes() {
        assert_eq!(limit_scale((8192, 2328), 8192), None);
        let scale = limit_scale((9313, 2328), 8192).unwrap();
        let (w, h) = scale_size((9313, 2328), scale);
        assert!(w.min(8192) == 8192 && h <= 2048, "{w}x{h}");
    }

    #[test]
    fn rotated_screens_need_more_height() {
        let landscape = fit_scale((6000, 4000), &[(2560, 1440)]).unwrap();
        let portrait = fit_scale((6000, 4000), &[(1440, 2560)]).unwrap();
        assert!(portrait > landscape);
    }

    #[test]
    fn small_images_are_left_alone() {
        assert_eq!(fit_scale((1920, 1080), &[(2560, 1440)]), None);
        assert_eq!(fit_scale((2560, 1440), &[(2560, 1440)]), None);
        assert_eq!(fit_scale((6000, 4000), &[]), None);
    }

    #[test]
    fn transitions_ease_from_zero_to_one() {
        let second = Duration::from_secs(1);
        assert_eq!(transition_progress(Duration::ZERO, second), 0.0);
        assert_eq!(transition_progress(second / 2, second), 0.5);
        assert_eq!(transition_progress(second, second), 1.0);
        assert_eq!(transition_progress(second * 2, second), 1.0);
        // Eased, so it starts slower than linear.
        assert!(transition_progress(second / 10, second) < 0.1);
    }

    #[test]
    fn zero_length_transitions_finish_at_once() {
        assert_eq!(transition_progress(Duration::ZERO, Duration::ZERO), 1.0);
    }

    #[test]
    fn shader_is_valid_and_uniforms_match() {
        let src = include_str!("shader.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));

        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (uniforms, _) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Uniforms"))
            .expect("shader declares Uniforms");
        assert_eq!(
            layouter[uniforms].size as usize,
            std::mem::size_of::<Uniforms>()
        );
    }
}
