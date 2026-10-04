//! Post-processing for MoGe-2, which predicts a 3D point per pixel rather
//! than a depth map. Its z values are only known up to an unknown shift, so
//! this recovers the shift the same way MoGe's own Python code does, then
//! reads depth off the shifted z.

/// MoGe's ViT token budget, matching its own default (resolution level 9).
pub const NUM_TOKENS: i64 = 3600;

/// Depth for every pixel from MoGe's `points` ([H, W, 3]) and `mask`
/// ([H, W]) outputs. Pixels the model marks as invalid, which is mostly
/// sky, get infinite depth so they end up furthest away.
pub fn depth_from_points(points: &[f32], mask: &[f32], width: usize, height: usize) -> Vec<f32> {
    let valid: Vec<bool> = mask.iter().map(|&m| m > 0.5).collect();
    let shift = recover_shift(points, &valid, width, height);
    (0..width * height)
        .map(|i| {
            let z = points[i * 3 + 2] + shift;
            if valid[i] && z > 0.0 {
                z
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

/// The z shift that best fits a pinhole camera centred on the image: the
/// one minimising |f * xy / (z + shift) - uv| over valid pixels, where the
/// best focal length f has a closed form for each shift. Like MoGe, it
/// works on a 64x64 sample of the map.
fn recover_shift(points: &[f32], valid: &[bool], width: usize, height: usize) -> f32 {
    const SAMPLES: usize = 64;
    let mut samples = Vec::with_capacity(SAMPLES * SAMPLES);
    for sy in 0..SAMPLES {
        for sx in 0..SAMPLES {
            let x = sx * width / SAMPLES;
            let y = sy * height / SAMPLES;
            let i = y * width + x;
            if valid[i] {
                let p = &points[i * 3..i * 3 + 3];
                samples.push((view_plane_uv(x, y, width, height), [p[0], p[1], p[2]]));
            }
        }
    }
    if samples.len() < 2 {
        return 0.0;
    }

    // Every shifted z must stay positive, so search shifts just above
    // -min(z), on a log scale relative to the spread of z.
    let (min_z, max_z) = samples
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), (_, p)| {
            (lo.min(p[2]), hi.max(p[2]))
        });
    let spread = (max_z - min_z).max(1e-6);
    let shift_at = |k: f64| -min_z + spread * 10f32.powf(k as f32);
    let error_at = |k: f64| fit_error(&samples, shift_at(k));

    // Coarse scan over offsets from 1/1000 to 1000 times the spread...
    let steps = 120;
    let ks: Vec<f64> = (0..=steps)
        .map(|i| -3.0 + 6.0 * i as f64 / steps as f64)
        .collect();
    let best = (0..ks.len())
        .min_by(|&a, &b| error_at(ks[a]).total_cmp(&error_at(ks[b])))
        .unwrap_or(0);

    // ...then a golden-section search between the best scan point's
    // neighbours.
    let (mut lo, mut hi) = (ks[best.saturating_sub(1)], ks[(best + 1).min(steps)]);
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let a = hi - ratio * (hi - lo);
        let b = lo + ratio * (hi - lo);
        if error_at(a) < error_at(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    shift_at((lo + hi) / 2.0)
}

/// Squared reprojection error for a shift, using the focal length that
/// minimises it.
fn fit_error(samples: &[([f32; 2], [f32; 3])], shift: f32) -> f64 {
    let projected: Vec<(f64, f64)> = samples
        .iter()
        .map(|(_, p)| {
            let z = f64::from(p[2] + shift);
            (f64::from(p[0]) / z, f64::from(p[1]) / z)
        })
        .collect();
    let (dot, norm) =
        samples
            .iter()
            .zip(&projected)
            .fold((0.0, 0.0), |(dot, norm), ((uv, _), (px, py))| {
                (
                    dot + px * f64::from(uv[0]) + py * f64::from(uv[1]),
                    norm + px * px + py * py,
                )
            });
    let focal = dot / norm.max(1e-12);
    samples
        .iter()
        .zip(&projected)
        .map(|((uv, _), (px, py))| {
            (focal * px - f64::from(uv[0])).powi(2) + (focal * py - f64::from(uv[1])).powi(2)
        })
        .sum()
}

/// MoGe's normalised image-plane coordinates for a pixel: centred on the
/// image, with the corners at plus or minus half the diagonal, which is 1.
fn view_plane_uv(x: usize, y: usize, width: usize, height: usize) -> [f32; 2] {
    let (w, h) = (width as f32, height as f32);
    let aspect = w / h;
    let diagonal = (1.0 + aspect * aspect).sqrt();
    let (span_x, span_y) = (aspect / diagonal, 1.0 / diagonal);
    [
        span_x * (2.0 * x as f32 + 1.0 - w) / w,
        span_y * (2.0 * y as f32 + 1.0 - h) / h,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point map as MoGe would predict it for a pinhole camera with the
    /// given focal length, with z offset by `-shift`.
    fn synthetic_points(width: usize, height: usize, focal: f32, shift: f32) -> Vec<f32> {
        let mut points = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let [u, v] = view_plane_uv(x, y, width, height);
                // A tilted plane, nearer at the bottom of the image.
                let depth = 3.0 - 1.5 * y as f32 / height as f32 + 0.3 * x as f32 / width as f32;
                points.extend([u * depth / focal, v * depth / focal, depth - shift]);
            }
        }
        points
    }

    #[test]
    fn recovers_a_known_shift() {
        let (w, h) = (96, 64);
        let points = synthetic_points(w, h, 1.3, 0.8);
        let shift = recover_shift(&points, &vec![true; w * h], w, h);
        assert!((shift - 0.8).abs() < 1e-2, "recovered shift {shift}");
    }

    #[test]
    fn invalid_pixels_are_infinitely_far() {
        let (w, h) = (96, 64);
        let points = synthetic_points(w, h, 1.3, 0.8);
        let mut mask = vec![1.0; w * h];
        mask[0] = 0.0;
        let depth = depth_from_points(&points, &mask, w, h);
        assert_eq!(depth[0], f32::INFINITY);
        // Valid pixels get their true depth back: 3.0 at the top-left corner
        // of the plane, give or take the solver's tolerance.
        assert!((depth[1] - 3.0).abs() < 2e-2, "depth {}", depth[1]);
    }

    #[test]
    fn view_plane_corners_sit_on_the_unit_diagonal() {
        let [u, v] = view_plane_uv(0, 0, 4, 3);
        // Pixel centres sit half a pixel in from the corner (-0.8, -0.6).
        assert!((u - -0.6).abs() < 1e-6 && (v - -0.4).abs() < 1e-6);
    }
}
