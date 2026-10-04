// Fullscreen depth-based parallax shader, with depth-ordered transitions
// between two wallpapers.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from vertex index, so no vertex buffers are needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
// The wallpaper being transitioned to. The same as the current one when
// no transition is running.
@group(0) @binding(4) var next_color_tex: texture_2d<f32>;
@group(0) @binding(5) var next_depth_tex: texture_2d<f32>;
// Each wallpaper's depth ranks, from `DepthMap::ranks`.
@group(0) @binding(6) var ranks: texture_1d<f32>;
@group(0) @binding(7) var next_ranks: texture_1d<f32>;

// Must match `Uniforms` in renderer.rs.
struct Uniforms {
    cursor_offset: vec2<f32>,
    intensity: f32,
    // 0 shows the current wallpaper, 1 the next one.
    progress: f32,
    // Fraction of each image shown on each axis after cropping to the
    // screen's aspect ratio.
    uv_scale: vec2<f32>,
    next_uv_scale: vec2<f32>,
    // 1 switches near pixels first, 0 far pixels first.
    near_first: f32,
    _pad: f32,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

// Zoom in 2.5% each side so displacement pulls in real pixels from beyond
// the visible area instead of stretching the edges.
const MARGIN: f32 = 0.025;
// Half the width of the slice of depth ranks that is part way through
// switching at any moment. Narrower gives a crisper wavefront, wider a
// softer dissolve.
const BAND: f32 = 0.08;

// Crop to the screen's aspect ratio and apply the margin.
fn crop(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    return 0.5 + (uv - 0.5) * scale * (1.0 - 2.0 * MARGIN);
}

fn load_depth(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let size = textureDimensions(tex);
    return textureLoad(tex, vec2<u32>(uv * vec2<f32>(size)), 0).r;
}

// The fraction of a wallpaper's pixels farther away than `depth`,
// interpolated from its 257 ranks.
fn depth_rank(table: texture_1d<f32>, depth: f32) -> f32 {
    let x = clamp(depth, 0.0, 1.0) * 256.0;
    let i = min(u32(x), 255u);
    let farther = textureLoad(table, i, 0).r;
    let next = textureLoad(table, i + 1u, 0).r;
    return mix(farther, next, x - f32(i));
}

// How far a pixel has switched to the next wallpaper. Think of a plane
// sweeping through both scenes, taking away the current wallpaper's
// surfaces and putting the next one's in place as it passes them, with
// each pixel showing whichever surface is nearest.
//
// Near first, a pixel switches once the plane reaches the next
// wallpaper's surface there, which is then in front of anything left of
// the current one. Far first, it also has to wait for the current
// wallpaper's surface to go, as that is in front until it does.
//
// Depths are compared by rank, so the sweep spends similar time on each
// part of the picture.
fn transition_weight(depth: f32, next_depth: f32, t: f32) -> f32 {
    let rank = depth_rank(ranks, depth);
    let next_rank = depth_rank(next_ranks, next_depth);
    // How far the plane has to travel before this pixel switches.
    let distance = select(max(rank, next_rank), 1.0 - next_rank, u.near_first > 0.5);
    // The plane travels from -BAND to 1 + BAND, so every pixel starts at 0
    // and ends at 1 whatever its depth.
    let plane = t * (1.0 + 2.0 * BAND) - BAND;
    return 1.0 - smoothstep(plane - BAND, plane + BAND, distance);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = crop(in.uv, u.uv_scale);
    // Depth comes from the same image position as the colour.
    var depth = load_depth(depth_tex, uv);
    if u.progress <= 0.0 {
        // Cursor right (+x) shifts near pixels left. Scaled by uv_scale so a
        // cropped axis moves the same amount on screen as an uncropped one.
        let shift = u.cursor_offset * depth * u.intensity;
        return textureSample(color_tex, tex_sampler, uv - shift * u.uv_scale);
    }

    let next_uv = crop(in.uv, u.next_uv_scale);
    let next_depth = load_depth(next_depth_tex, next_uv);
    let weight = transition_weight(depth, next_depth, u.progress);
    // Blending depth as well as colour morphs the parallax geometry, and
    // both images shift by it so they move together.
    depth = mix(depth, next_depth, weight);
    let shift = u.cursor_offset * depth * u.intensity;
    let color = textureSample(color_tex, tex_sampler, uv - shift * u.uv_scale);
    let next_color = textureSample(next_color_tex, tex_sampler, next_uv - shift * u.next_uv_scale);
    return mix(color, next_color, weight);
}
