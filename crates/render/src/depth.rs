use anyhow::{Context, Result};
use image::imageops::{self, FilterType};
use std::path::Path;

/// Depth map at the wallpaper's native resolution.
/// Values are u16 normalized, from 0 (the farthest) to 1 (the nearest).
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
    /// The map as 16-bit floats from 0 to 1, as the GPU takes it.
    pub fn to_f16(&self) -> Vec<half::f16> {
        (self.data.iter())
            .map(|&d| half::f16::from_f32(f32::from(d) / f32::from(u16::MAX)))
            .collect()
    }

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
    pub fn height_ranks(&self, ground: &Ground) -> Vec<f32> {
        let mut counts = [0u32; RANK_POINTS - 1];
        let width = self.width.max(1) as usize;
        for (i, &d) in self.data.iter().enumerate() {
            let x = ((i % width) as f32 + 0.5) / width as f32;
            let y = ((i / width) as f32 + 0.5) / self.height as f32;
            let height = tide_height([x, y], f32::from(d) / 65535.0, ground);
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

/// How close a depth of 0 is taken to be, which keeps the farthest pixels
/// at a finite distance. Must match TIDE_NEAR in shader.wgsl.
pub const TIDE_NEAR: f32 = 0.1;
/// The camera's field of view, which the depth map doesn't record, so the
/// tide assumes a typical photo's: the height of the view at a distance
/// of 1, which is about 45 degrees top to bottom.
const TIDE_VIEW: f32 = 0.83;
/// How many points across and down `DepthMap::ground` looks at.
const GROUND_SAMPLES: usize = 64;
/// How many planes `DepthMap::ground` tries.
const GROUND_TRIES: usize = 400;
/// How far a point can be from a plane and still count as on it, as a
/// fraction of the point's distance.
const GROUND_TOLERANCE: f32 = 0.03;
/// How far ground that faces the camera can lean to one side, in degrees.
const GROUND_LEAN: f32 = 40.0;
/// How different in angle another plane has to be, in degrees, to count
/// as a second slope.
const GROUND_SLOPES: f32 = 15.0;

/// The ground the tide rises from, as the four numbers `tide_height` needs.
/// Must match how tide_height in shader.wgsl uses them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground(pub [f32; 4]);

impl Ground {
    /// For a level camera with the horizon across the middle of the image,
    /// when there's no convincing ground.
    pub const LEVEL: Self = Self([0.0, -1.0, 0.5, 0.0]);

    /// The plane through the scene where `normal` dotted with a point,
    /// plus `offset`, is 0, with the camera on the side `normal` points to.
    /// Heights are measured from it in units of how high the camera is
    /// above it.
    fn from_plane(normal: [f32; 3], offset: f32, aspect: f32) -> Self {
        // A point at image (x, y) and distance z is z * (along x, along y,
        // 1), so its height is z * (a * x + b * y + c) + 1.
        let across = TIDE_VIEW * aspect;
        let [nx, ny, nz] = normal;
        Self([
            nx * across / offset,
            -ny * TIDE_VIEW / offset,
            (nz - 0.5 * nx * across + 0.5 * ny * TIDE_VIEW) / offset,
            1.0,
        ])
    }
}

/// Roughly how high a point is in the scene above the ground, squashed
/// into (-1, 1), from where it is in the image and its depth. The model's
/// depth has no true scale and the camera's field of view is a guess, so
/// this is only rough, but good enough for water to find its level. Must
/// match tide_height in shader.wgsl.
pub fn tide_height([x, y]: [f32; 2], depth: f32, ground: &Ground) -> f32 {
    let distance = 1.0 / (depth + TIDE_NEAR);
    let [a, b, c, d] = ground.0;
    let height = distance * (a * x + b * y + c) + d;
    height / (1.0 + height.abs())
}

impl DepthMap {
    /// The biggest flat surface in the scene that faces up or towards the
    /// camera, which the tide takes to be the ground, or `Ground::LEVEL`
    /// if it isn't convincing. See `convincing`.
    pub fn ground(&self) -> Ground {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let points = self.scene_points(aspect);
        let Some((mut plane, _)) = strongest_plane(&points) else {
            return Ground::LEVEL;
        };
        // Fit the plane to the points on it, then again to the points on
        // that.
        for _ in 0..2 {
            let on_it: Vec<[f32; 3]> = points.iter().filter(|p| on(plane, p)).copied().collect();
            plane = fit_plane(&on_it, plane.0).unwrap_or(plane);
        }
        if !convincing(plane, &points) {
            return Ground::LEVEL;
        }
        Ground::from_plane(plane.0, plane.1, aspect)
    }

    /// Where a grid of GROUND_SAMPLES points across and down the map are in
    /// the scene, leaving out any at a depth of 0, like sky. x is right, y
    /// up and z away from the camera, at the scale `tide_height` uses.
    fn scene_points(&self, aspect: f32) -> Vec<[f32; 3]> {
        let (width, height) = (self.width as usize, self.height as usize);
        let mut points = Vec::with_capacity(GROUND_SAMPLES * GROUND_SAMPLES);
        for row in 0..GROUND_SAMPLES {
            for col in 0..GROUND_SAMPLES {
                let px = (col * 2 + 1) * width / (GROUND_SAMPLES * 2);
                let py = (row * 2 + 1) * height / (GROUND_SAMPLES * 2);
                let Some(&d) = self.data.get(py * width + px) else {
                    continue;
                };
                if d == 0 {
                    continue;
                }
                let x = (px as f32 + 0.5) / width as f32;
                let y = (py as f32 + 0.5) / height as f32;
                let z = 1.0 / (f32::from(d) / 65535.0 + TIDE_NEAR);
                points.push([
                    z * (x - 0.5) * TIDE_VIEW * aspect,
                    z * (0.5 - y) * TIDE_VIEW,
                    z,
                ]);
            }
        }
        points
    }
}

/// A plane as a unit normal and offset, where the normal dotted with a
/// point, plus the offset, is 0.
type Plane = ([f32; 3], f32);

/// Whether `point` is on `plane`, within GROUND_TOLERANCE.
fn on((normal, offset): Plane, point: &[f32; 3]) -> bool {
    (dot(normal, *point) + offset).abs() < GROUND_TOLERANCE * point[2]
}

/// The plane that could be the ground with the most `points` on it, and
/// how many, from planes through random sets of three points.
fn strongest_plane(points: &[[f32; 3]]) -> Option<(Plane, usize)> {
    if points.len() < 3 {
        return None;
    }
    let mut random = 0x9e37_79b9_u32;
    let mut pick = || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        points[random as usize % points.len()]
    };
    let mut best = None;
    let mut most = 0;
    for _ in 0..GROUND_TRIES {
        let [a, b, c] = [pick(), pick(), pick()];
        let Some(plane) = plane_facing_camera(cross(sub(b, a), sub(c, a)), a) else {
            continue;
        };
        let count = points.iter().filter(|p| on(plane, p)).count();
        if count > most {
            most = count;
            best = Some((plane, count));
        }
    }
    best
}

/// Whether `plane`, the strongest in the scene, is convincing as the
/// ground. Close-ups and abstract images often have none, and their
/// strongest plane faces the camera but leans well to one side. Ground
/// that faces up doesn't count if there's another plane nearly as strong
/// at quite a different angle, like the two sides of a valley.
fn convincing(plane: Plane, points: &[[f32; 3]]) -> bool {
    let [nx, ny, nz] = plane.0;
    if ny < nx.abs() {
        return -nz >= GROUND_LEAN.to_radians().cos();
    }
    let rest: Vec<[f32; 3]> = points.iter().filter(|p| !on(plane, p)).copied().collect();
    let Some((other, others)) = strongest_plane(&rest) else {
        return true;
    };
    let count = points.len() - rest.len();
    others * 2 < count || dot(plane.0, other.0) > GROUND_SLOPES.to_radians().cos()
}

/// The plane through `point` facing along `normal`, as a unit normal and
/// offset, flipped so the camera is on the side it faces. None if it can't
/// be the ground: if it passes through the camera, faces down the picture
/// like a ceiling, or faces more sideways than up and isn't within about
/// 45 degrees of facing the camera either, like a wall to one side.
fn plane_facing_camera(normal: [f32; 3], point: [f32; 3]) -> Option<Plane> {
    let length = dot(normal, normal).sqrt();
    if length < 1e-9 {
        return None;
    }
    let mut normal = normal.map(|n| n / length);
    let mut offset = -dot(normal, point);
    if offset < 0.0 {
        normal = normal.map(|n| -n);
        offset = -offset;
    }
    let [nx, ny, nz] = normal;
    let faces_up = ny >= nx.abs();
    let faces_camera = -nz >= 0.7;
    (offset > 1e-4 && (faces_up || faces_camera)).then_some((normal, offset))
}

/// The plane closest to all of `points`, through their middle and facing
/// the way they spread least, starting the search from `normal`.
fn fit_plane(points: &[[f32; 3]], mut normal: [f32; 3]) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f32;
    let middle = points
        .iter()
        .fold([0.0; 3], |sum, p| {
            [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]]
        })
        .map(|s| s / n);
    let mut spread = [[0.0f32; 3]; 3];
    for p in points {
        let d = sub(*p, middle);
        for i in 0..3 {
            for j in 0..3 {
                spread[i][j] += d[i] * d[j];
            }
        }
    }
    // The way the points spread least is the way they spread most once
    // the spread is turned inside out, which repeated multiplying finds.
    let total = spread[0][0] + spread[1][1] + spread[2][2];
    for _ in 0..50 {
        let next: [f32; 3] = std::array::from_fn(|i| total * normal[i] - dot(spread[i], normal));
        let length = dot(next, next).sqrt();
        if length < 1e-12 {
            return None;
        }
        normal = next.map(|v| v / length);
    }
    plane_facing_camera(normal, middle)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

    /// A map of flat ground facing along `normal`, with the camera 1 above
    /// it, and sky wherever the camera can't see the ground.
    fn ground_map(normal: [f32; 3]) -> DepthMap {
        surfaces_map(&[normal])
    }

    /// Like `ground_map`, but each pixel sees the nearest of several
    /// surfaces, each 1 from the camera.
    fn surfaces_map(normals: &[[f32; 3]]) -> DepthMap {
        let (width, height) = (96, 64);
        let aspect = width as f32 / height as f32;
        let data = (0..width * height)
            .map(|i| {
                let x = ((i % width) as f32 + 0.5) / width as f32;
                let y = ((i / width) as f32 + 0.5) / height as f32;
                let ray = [(x - 0.5) * TIDE_VIEW * aspect, (0.5 - y) * TIDE_VIEW, 1.0];
                let distance = normals
                    .iter()
                    .map(|&normal| -1.0 / dot(normal, ray))
                    .filter(|&d| d > 0.0)
                    .fold(f32::INFINITY, f32::min);
                let depth = 1.0 / distance - TIDE_NEAR;
                if distance > 0.0 && depth > 0.0 {
                    (depth.min(1.0) * 65535.0).round() as u16
                } else {
                    0
                }
            })
            .collect();
        DepthMap {
            data,
            width: width as u32,
            height: height as u32,
        }
    }

    /// The normal of level ground seen by a camera pitched down and rolled
    /// by these many degrees.
    fn tilt(pitch: f32, roll: f32) -> [f32; 3] {
        let (pitch, roll) = (pitch.to_radians(), roll.to_radians());
        [
            roll.sin() * pitch.cos(),
            roll.cos() * pitch.cos(),
            -pitch.sin(),
        ]
    }

    #[test]
    fn ground_is_found_however_the_camera_points() {
        for (name, normal) in [
            ("level", tilt(0.0, 0.0)),
            ("looking down and tilted", tilt(30.0, 20.0)),
            ("from above", tilt(80.0, 0.0)),
        ] {
            let map = ground_map(normal);
            let ground = map.ground();
            let width = map.width as usize;
            for (i, &d) in map.data.iter().enumerate().filter(|(_, d)| **d > 0) {
                let x = ((i % width) as f32 + 0.5) / width as f32;
                let y = ((i / width) as f32 + 0.5) / map.height as f32;
                let depth = f32::from(d) / 65535.0;
                let on = tide_height([x, y], depth, &ground);
                assert!(on.abs() < 0.02, "{name}: the ground is at {on}");
                // Something nearer than the ground there stands on it.
                let above = tide_height([x, y], depth + 0.05, &ground);
                assert!(above > on, "{name}: {above} isn't above {on}");
            }
        }
    }

    #[test]
    fn no_convincing_ground_is_level() {
        // The two sides of a valley.
        let valley = surfaces_map(&[tilt(0.0, 25.0), tilt(0.0, -25.0)]);
        assert_eq!(valley.ground(), Ground::LEVEL);
        // Something facing the camera but leaning well to one side.
        let leaning = ground_map(tilt(90.0 - 43.0, 90.0));
        assert_eq!(leaning.ground(), Ground::LEVEL);
    }

    #[test]
    fn sky_is_highest() {
        let map = ground_map([0.0, 1.0, 0.0]);
        let ground = map.ground();
        let sky = tide_height([0.5, 0.1], 0.0, &ground);
        let near = tide_height([0.5, 0.9], 0.9, &ground);
        assert!(sky > 0.5 && sky > near, "{sky}, {near}");
    }

    #[test]
    fn height_ranks_run_from_zero_to_one() {
        let map = DepthMap {
            data: vec![0, 20000, 40000, 65535, 0, 20000, 40000, 65535],
            width: 4,
            height: 2,
        };
        let ranks = map.height_ranks(&Ground::LEVEL);
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
